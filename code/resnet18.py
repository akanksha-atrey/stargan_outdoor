from torchvision import transforms, models, datasets
from PIL import Image
import os
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import argparse
import numpy as np
from __init__ import LANDMARK_DATA, WC_DATA, TRANSIENT_DATA, MODEL_PATH

torch.set_default_dtype(torch.float64)
dtype = torch.float64
print_every = 100


def load_dataset(mode, img_dir):

	"""
	Loads the dataset and transforms it to conform to the ResNet specifications. 
	Size : 224*224
	Normalization : mean : [0.485, 0.456, 0.406]
					std : [0.229, 0.224, 0.225]

	Arguments:
		mode : 'train' or 'test'
		img_dir : Location of dataset

	Returns:
		data_loader (DataLoader) :  object with the training or test data
	"""
	img_transforms = transforms.Compose([transforms.Resize((224, 224)),
										transforms.ToTensor(),
                                      	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
										])

	if mode=='train':
		train_data = datasets.ImageFolder(img_dir, transform=img_transforms)
		val_data = datasets.ImageFolder(img_dir, transform=img_transforms)
		
		val_size = 0.1
		num_train = len(train_data)
		indices = list(range(num_train))
		split = int(np.floor(val_size * num_train))

		np.random.seed(0)
		np.random.shuffle(indices)

		train_idx, val_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		val_sampler = SubsetRandomSampler(val_idx)

		train_loader = DataLoader(train_data, batch_size=16, sampler=train_sampler)
		val_loader = DataLoader(val_data, batch_size=16, sampler=val_sampler)
		return train_loader, val_loader

	else: 	
		test_data = datasets.ImageFolder(img_dir, transform=img_transforms)
		test_loader = DataLoader(test_data)
		return test_loader



def define_model(pretrained, num_classes):

	"""
	Define the ResNet model

	Arguments:
		pretrained (bool) : Whether a pretrained model should be used or not
		num_classes (int) : Number of classes in the dataset

	Returns:
		model (ResNet) : Defined model	
	"""

	model = models.resnet18(pretrained=pretrained)
	count = 0
	# Freeze the layers 1-3 of the ResNet, and train the rest
	if pretrained:
		for layer in model.children():
			count += 1
			if count < 4:
				for param in layer.parameters():
					param.requires_grad = False

	# Changing FC layer to classify our dataset
	model.fc = nn.Sequential(
				nn.Linear(in_features=512, out_features=num_classes)
				)
	return model



def train(model, dataset_name, train_loader, val_loader, epochs, lr=0.003):

	"""
	Train the model on our dataset

	Arguments:
		model(ResNet) : The defined model
		dataset_name : Which dataset is being used
		train_loader(DataLoader) : The training data
		val_loader(DataLoader) : The validation data
		epochs(int) : Number of training epochs

	Returns:
		Nothing, but prints accuracies and saves model
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	optimizer = optim.Adam(model.parameters(), lr)
	highest_val_accuracy = 0.0
	for e in range(epochs):
		for t, (x, y) in enumerate(train_loader):
			model.train()  # put model to training mode
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			scores = model(x)
			loss = F.cross_entropy(scores, y)

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, loss = %.4f' % (t, loss.item()))
				print()
		val_acc = check_accuracy('train', model, val_loader)
		if val_acc > highest_val_accuracy:
			model_file = os.path.join(MODEL_PATH, dataset_name+'.pth')
			torch.save(model, model_file)


def check_accuracy(mode, model, dataloader):

	"""
	Check accuracy

	Arguments:
		mode(str) : 'train' or 'test'
		model(ResNet) : trained model
		dataloader(DataLoader) : Either the validation or test data

	Returns:
		acc : Classification accuracy
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if mode=='train':
		print('Checking accuracy on validation set')
	else:
		print('Checking accuracy on test set')   
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	with torch.no_grad():
		for x, y in dataloader:
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
	return acc

def test(model_dir, dataloader):
	
	"""
	Test the model

	Arguments:
		model_dir : Path where best model was stored
		dataloader(DataLoader) : Test data
	Returns:
		Nothing, but checks accuracy and prints it

	"""

	model = torch.load(model_dir)
	check_accuracy('test', model, dataloader)

if __name__ == '__main__':

	if not os.path.exists(MODEL_PATH):
		os.mkdir(MODEL_PATH)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='train', help='Run the model in this mode', type=str)
	parser.add_argument('--n_class', help='Number of classes in the dataset', type=int)
	parser.add_argument('--dataset', help='Name of dataset. Can be world_cities, landmarks or transient', type=str)
	parser.add_argument('--pretrained', help="If we want a pretrained network or not" , type=bool)
	parser.add_argument('--lr', help='learning rate', type=float)
	parser.add_argument('--epochs', type=int)
	config = parser.parse_args()

	dataset_path = {'world_cities': WC_DATA, 'landmarks' : LANDMARK_DATA, 'transient' : TRANSIENT_DATA }
	dataset = config.dataset
	
	if config.mode=='train':
		model = define_model(config.pretrained, config.n_class)
		train_data_path = os.path.join(dataset_path[dataset], 'train')
		train_loader, val_loader = load_dataset(config.mode, train_data_path)
		train(model, dataset, train_loader, val_loader, config.epochs, config.lr)

	else:
		test_data_path = os.path.join(dataset_path[dataset], 'test')
		test_loader = load_dataset(config.mode, test_data_path)

		trained_model_path = os.path.join(MODEL_PATH, dataset+'.pth')
		test(trained_model_path, test_loader)

		# python resnet18.py --mode 'train' --n_class 2 --dataset 'world_cities' --pretrained True --lr 0.004 --epochs 1

	