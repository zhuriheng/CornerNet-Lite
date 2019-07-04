#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import queue
import pprint
import random
import shutil
import time
import argparse
import importlib
import threading
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from py3nvml.py3nvml import *
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.test import test_func
from core.utils import stdout_to_tqdm, pLogger
from core.config import SystemConfig
from core.sample import data_sampling_func
from core.nnet.py_factory import NetworkFactory

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--initialize", action="store_true")

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="number of nodes of distributed training")
    parser.add_argument("--rank", default=0, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist-url", default=None, type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str)

    args = parser.parse_args()
    return args

def prefetch_data(train_logger, system_config, db, queue, sample_data, data_aug):
    ind = 0
    train_logger.train_logging("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(system_config, db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def _pin_memory(ts):
    if type(ts) is list:
        return [t.pin_memory() for t in ts]
    return ts.pin_memory()

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [_pin_memory(x) for x in data["xs"]]
        data["ys"] = [_pin_memory(y) for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(train_logger, system_config, dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(train_logger, system_config, db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def terminate_tasks(tasks):
    for task in tasks:
        task.terminate()

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def train(train_logger, training_dbs, validation_db, system_config, model, args):
    # reading arguments from command
    start_iter  = args.start_iter
    distributed = args.distributed
    world_size  = args.world_size
    initialize  = args.initialize
    gpu         = args.gpu
    rank        = args.rank

    # reading arguments from json file
    batch_size       = system_config.batch_size
    learning_rate    = system_config.learning_rate
    max_iteration    = system_config.max_iter
    pretrained_model = system_config.pretrain
    snapshot         = system_config.snapshot
    val_iter         = system_config.val_iter
    display          = system_config.display
    decay_rate       = system_config.decay_rate
    stepsize         = system_config.stepsize

    train_logger.train_logging("Process {}: building model...".format(rank))
    nnet = NetworkFactory(system_config, model, distributed=distributed, gpu=gpu)
    if initialize:
        nnet.save_params(0)
        exit(0)

    # queues storing data for training
    training_queue   = Queue(system_config.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_config.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # allocating resources for parallel reading
    training_tasks = init_parallel_jobs(train_logger, system_config, training_dbs, training_queue, data_sampling_func, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(train_logger, system_config, [validation_db], validation_queue, data_sampling_func, False)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        train_logger.train_logging("Process {}: loading from pretrained model".format(rank))
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        nnet.load_params(start_iter)
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.set_lr(learning_rate)
        train_logger.train_logging("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    if rank == 0:
        train_logger.train_logging("training start...")
    nnet.cuda()
    nnet.train_mode()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)

            train_logger.tb_logging('Train/loss', {'tloss': training_loss.item()}, iteration)

            if display and iteration % display == 0:
                train_logger.train_logging("Process {}: training loss at iteration {}: {}".format(rank, iteration, training_loss.item()))
            del training_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                # calculate validation loss
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                train_logger.train_logging("Process {}: validation loss at iteration {}: {}".format(rank, iteration, validation_loss.item()))
                train_logger.tb_logging('Val/loss', {'vloss': validation_loss.item()}, iteration)
                nnet.train_mode()

            if iteration % snapshot == 0 and rank == 0:
                nnet.eval_mode()
                # calculate validation mAP
                val_split = system_config.val_split
                mAP, _, detect_average_time = test(validation_db, system_config, nnet, val_iter, val_split, debug=True)
                train_logger.train_logging("Process {}: mAP at iteration {}: {}".format(rank, iteration, mAP))
                train_logger.train_logging("Detect average time: {}".format(detect_average_time))
                nnet.train_mode()

            if iteration % snapshot == 0 and rank == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

            # dc = 0
            # handle = nvmlDeviceGetHandleByIndex(dc)
            # res = nvmlDeviceGetUtilizationRates(handle)
            # gpu_util = res.gpu
            # res = nvmlDeviceGetMemoryInfo(handle)
            # gpu_mem = res.used / 1024 / 1024
            # train_logger.tb_logging('data/NV', {'gpu-util': gpu_util, 'gpu-mem': gpu_mem}, iteration)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    terminate_tasks(training_tasks)
    terminate_tasks(validation_tasks)

def test(db, system_config, nnet, val_iter, split, debug=False, suffix=None):
    split    = split
    testiter = val_iter
    debug    = debug
    suffix   = suffix

    result_dir = system_config.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])

    a, b, detect_average_time = test_func(system_config, db, nnet, result_dir, debug=debug)
    return a, b, detect_average_time

def main(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    rank = args.rank

    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)

    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])

    model_file  = "core.models.{}".format(args.cfg_file)
    model_file  = importlib.import_module(model_file)
    model       = model_file.model(num_classes=config["db"]["categories"])

    train_split = system_config.train_split
    val_split   = system_config.val_split

    ckpt_path = os.path.join('cache/nnet/', args.cfg_file, date)
    train_logger = pLogger(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.join(ckpt_path))
    shutil.copyfile('{}'.format(cfg_file), '{}/{}'.format(ckpt_path, args.cfg_file + ".json"))

    train_logger.train_logging("Process {}: loading all datasets...".format(rank))
    dataset = system_config.dataset
    workers = args.workers
    train_logger.train_logging("Process {}: using {} workers".format(rank, workers))
    training_dbs = [datasets[dataset](config["db"], split=train_split, sys_config=system_config) for _ in range(workers)]
    validation_db = datasets[dataset](config["db"], split=val_split, sys_config=system_config)

    if rank == 0:
        print("system config...")
        pprint.pprint(system_config.full)
        train_logger.train_logging("system config...")
        train_logger.train_logging(system_config.full)

        print("db config...")
        pprint.pprint(training_dbs[0].configs)
        train_logger.train_logging("db config...")
        train_logger.train_logging(training_dbs[0].configs)

        train_logger.train_logging("len of db: {}".format(len(training_dbs[0].db_inds)))
        train_logger.train_logging("distributed: {}".format(args.distributed))

    train(train_logger, training_dbs, validation_db, system_config, model, args)

if __name__ == "__main__":
    args = parse_args()

    distributed = args.distributed
    world_size  = args.world_size

    if distributed and world_size < 0:
        raise ValueError("world size must be greater than 0 in distributed training")

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(None, ngpus_per_node, args)
