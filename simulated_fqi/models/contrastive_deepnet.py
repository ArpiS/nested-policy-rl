import os, sys
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision import datasets, models, transforms
from numpy import linalg as LA


class ContrastiveDataset(Dataset):
	def __init__(self, tuples):
		self.tuples = tuples

		def construct_pairs(tuples):
			X = []
			y = []
			for t in self.tuples:
				if t[4] == 'background':
					one_hot_a = [0] * 25
					s = t[0]
					a = t[1]
					one_hot_a[a] = 1
					r = t[3][0]
					if r == 0:
						r = 0.00000001

					blank_s = [0] * 46
					blank_a = [0]
					s_a = np.hstack((s, a, blank_s, blank_a))
					X.append(s_a)
					y.append(r)
				else:
					one_hot_a = [0] * 25
					s = t[0]
					a = t[1]
					one_hot_a[a] = 1
					r = t[3][0]
					if r == 0:
						r = 0.00000001

					s_a = np.hstack((s, a, s, a))
					X.append(s_a)
					y.append(r)
			return X, y

		self.X, self.y = construct_pairs(self.tuples)

	def __len__(self):
		return len(self.tuples)

	def __getitem__(self, idx):
		return (self.X[idx], self.y[idx])


class LinearContrastiveNet(nn.Module):
	def __init__(self):
		super(LinearContrastiveNet, self).__init__()
		self.name = "LinearContrastiveNet"
		self.fc1 = nn.Linear(94, 10)
		self.fc2 = nn.Linear(10, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = torch.sigmoid(x)
		return x


class ConvContrastiveNet(nn.Module):
	def __init__(self):
		super(ConvContrastiveNet, self).__init__()
		self.name = "ConvContrastiveNet"
		self.conv1 = self.cnn_apt_1 = nn.Conv1d(1, 10, 5)
		self.fc1 = nn.Linear(90, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.fc1(x)
		x = torch.sigmoid(x)
		return x


def weights_init(m):
	if isinstance(m, nn.Conv1d):
		nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(m.bias.data)
	if isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight.data, nonlinearity='sigmoid')
		nn.init.zeros_(m.bias.data)


class ContrastiveNet():
	def __init__(self, model, num_epochs=2):
		if model == 'convnet':
			self.model = ConvContrastiveNet()
		elif model == 'linnet':
			self.model = LinearContrastiveNet()
		else:
			raise Exception("Model must be convnet or linnet")

		model.apply(weights_init)
		self.num_epochs = num_epochs

	def fit(self, X, y):
		self.model.train()
		optimizer = SGD(self.model.parameters(), lr=0.001)
		criterion = nn.MSELoss()
		for epoch in range(self.num_epochs):
			for i, (s_a, r) in zip(X, y):
				if self.model.name == "ConvContrastiveNet":
					s_a = np.reshape(s_a, (1, 1, 94))
				s_a = torch.Tensor(s_a)
				r = torch.Tensor([r])
				pred_r = self.model(s_a)
				train_loss = criterion(pred_r, r)
				train_loss.backward()
				optimizer.step()

	def predict(self, X, y):
		self.model.eval()
		preds = []
		for (s_a, r) in zip(X, y):
			if self.model.name == "ConvContrastiveNet":
				s_a = np.reshape(s_a, (1, 1, 94))
			s_a = torch.Tensor(s_a)
			pred_r = self.model(s_a)
			preds.append(pred_r.detach().cpu().numpy().item())
		return preds
