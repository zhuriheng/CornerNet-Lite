import logging
import datetime
from tensorboardX import SummaryWriter
import os


def logger_init(log_filename):
	logger = logging.getLogger('Poseidon')
	logger.setLevel(level=logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(message)s')

	handler = logging.FileHandler(log_filename)
	handler.setLevel(level=logging.INFO)
	handler.setFormatter(formatter)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(formatter)

	logger.addHandler(handler)
	logger.addHandler(console)
	return logger


class pLogger(object):
	def __init__(self, log_path):
		# logger
		if not os.path.exists(log_path):
			os.makedirs(log_path)
		self.log_path = log_path
		self.logger = logger_init(os.path.join(log_path, 'log_file'))
		# tensorboard logger
		self.tb_logger = SummaryWriter(os.path.join(log_path, 'tb_file'))
		# train logger
		self.train_logger = logging.getLogger('Poseidon.Train')
		# sys logger
		self.sys_logger = logging.getLogger('Poseidon.Sys')

	def sys_logging(self, logging_string):
		self.sys_logger.info(logging_string)

	def train_logging(self, logging_string):
		self.train_logger.info(logging_string)

	def tb_logging(self, node_string, val, idx):
		self.tb_logger.add_scalars(node_string, val, idx)

	def tb_graph_logging(self, model, input_data):
		self.tb_logger.add_graph(model, (input_data,))

	def tb_figure_logging(self, node_string, figure, idx):
		self.tb_logger.add_figure(node_string, figure, idx)

	def tb_histogram_logging(self, model, idx):
		for name, param in model.named_parameters():
			layer, attr = os.path.splitext(name)
			attr = attr[1:]
			self.tb_logger.add_histogram("{}/{}_data".format(layer, attr), param, idx)
			self.tb_logger.add_histogram("{}/{}_grad".format(layer, attr), param.grad, idx)

	def close_logger(self):
		self.tb_logger.export_scalars_to_json(os.path.join(self.log_path, 'tensorboard_scalars.json'))
		self.tb_logger.close()


if __name__ == '__main__':
	import torch
	import torchvision
	train_logger = pLogger('usr')
	train_logger.sys_logging('this is syslog')
	train_logger.train_logging('this is syslog')
	train_logger.tb_logging('syslog', {"id": 13}, 1)
	train_logger.tb_logging('syslog', {"id": 14}, 2)
	train_logger.tb_logging('syslog', {"id": 15}, 3)

	input_data = torch.rand(16, 3, 224, 224)
	model = torchvision.models.resnet18()
	train_logger.tb_graph_logging(model, input_data)

	train_logger.close_logger()
