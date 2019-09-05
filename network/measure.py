# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#using this script to get flops of CNNs model

import torch
from torch.autograd import Variable
from functools import reduce
import operator

__all__ = ['measure_model', 'accuracy', 'draw_classification']

count_ops = 0
conv_ops = 0
count_params = 0


def get_num_gen(gen):
	return sum(1 for x in gen)


def is_leaf(model):
	return get_num_gen(model.children()) == 0


def get_layer_info(layer):
	layer_str = str(layer)
	type_name = layer_str[:layer_str.find('(')].strip()
	return type_name


def get_layer_param(model):
	return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
	global count_ops, conv_ops, count_params
	delta_ops = 0
	delta_params = 0
	multi_add = 1
	type_name = get_layer_info(layer)

	### ops_conv
	if type_name in ['Conv2d']:
		out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
		            layer.stride[0] + 1)
		out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
		            layer.stride[1] + 1)
		delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
		            layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
		# print (str(layer), delta_ops)
		conv_ops += delta_ops
		delta_params = get_layer_param(layer)

	### ops_nonlinearity
	elif type_name in ['ReLU', 'Sigmoid', 'PReLU', 'ReLU6']:
		delta_ops = x.numel()
		delta_params = get_layer_param(layer)

	### ops_pooling
	elif type_name in ['AvgPool2d', 'MaxPool2d']:
		in_w = x.size()[2]
		kernel_ops = layer.kernel_size * layer.kernel_size
		out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
		out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
		delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
		delta_params = get_layer_param(layer)

	elif type_name in ['AdaptiveAvgPool2d', 'AdaptiveMaxPool2d']:
		delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
		delta_params = get_layer_param(layer)

	### ops_linear
	elif type_name in ['Linear']:
		weight_ops = layer.weight.numel() * multi_add
		bias_ops = layer.bias.numel()
		delta_ops = x.size()[0] * (weight_ops + bias_ops)
		delta_params = get_layer_param(layer)

	### ops_nothing
	elif type_name in ['InstanceNorm2d', 'BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'Sequential',
	                   'ChannelShuffle']:
		delta_params = get_layer_param(layer)

	### unknown layer type
	else:
		raise TypeError('unknown layer type: %s' % type_name)

	count_ops += delta_ops
	count_params += delta_params
	return


def measure_model(model, H, W):
	global count_ops, conv_ops, count_params
	model.eval()
	count_ops = 0
	conv_ops = 0
	count_params = 0
	data = Variable(torch.zeros(1, 3, H, W))

	def should_measure(x):
		return is_leaf(x)

	def modify_forward(model):
		for child in model.children():
			if should_measure(child):
				def new_forward(m):
					def lambda_forward(x):
						measure_layer(m, x)
						return m.old_forward(x)

					return lambda_forward

				child.old_forward = child.forward
				child.forward = new_forward(child)
			else:
				modify_forward(child)

	def restore_forward(model):
		for child in model.children():
			# leaf node
			if is_leaf(child) and hasattr(child, 'old_forward'):
				child.forward = child.old_forward
				child.old_forward = None
			else:
				restore_forward(child)

	modify_forward(model)
	model.forward(data)
	restore_forward(model)

	return count_ops, conv_ops, count_params


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		if len(topk) == 1:
			k = topk[0]
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res = correct_k.mul_(100.0 / (batch_size + 1e-6))
		else:
			res = []
			for k in topk:
				correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / (batch_size + 1e-6)))
		return res


def draw_classification(img, output, label_csv):
	with open(label_csv) as f:
		labels = f.readlines()
	assert (len(labels) == len(output.flatten()))
	import cv2
	score, pred = output.topk(1, 1, True, True)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, str(labels[pred].strip()), (0, 20), font, 0.8, (0, 255, 0), 1)
	cv2.putText(img, str(round(score.item(), 2)), (0, 60), font, 0.8, (0, 255, 0), 1)

	return img

