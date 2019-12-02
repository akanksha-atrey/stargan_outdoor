import numpy as np
from PIL import Image
import os
import argparse
 
# This script converts images outputted by the stargan model into individual test images to be used as input to the resnet during classification.
# Once this script is run, the single images are stored as './results_resnet/dataset/attribute/img_name'

# This script has to be used as : generate_test_imgs.py --dataset landmarks OR generate_test_imgs.py --dataset world_cities OR generate_test_imgs.py --dataset transient

landmarks = ['original', 'EdinburghCastle', 'EiffelTower', 'GoldenGateBridge', 'GrandCanyon', 'Masada', 'MountRainier', 'NiagaraFalls']
world_cities = ['original', 'Amsterdam','Athens','Beijing', 'NewYork', 'Paris']
transient = ['original', 'autumn','clouds','dawndusk','daylight','fog','night','rain','snow','spring','storm','summer','sunny','sunrisesunset', 'winter']

dataset_names = {'landmarks': landmarks, 'world_cities': world_cities, 'transient':transient}

res_for_resnet_dir = './results_resnet'

img_count = 0

'''
Get single images for each consolidated image and saved in their respective class folder.

Arguments:
	dataset (str): dataset to be working on
	results_dir (str): the results directory which contain the output of StarGAN
	image (numpy array): image represented as a numpy array
'''
def get_single_imgs(dataset, results_dir, image, windowsize_r=128, windowsize_c=128):
	global img_count
	global dataset_names
	global res_for_resnet_dir

	res_dir = os.path.join(res_for_resnet_dir, dataset)
	if not os.path.exists(res_dir):
		os.mkdir(res_dir)

	# Crop out the window
	for r in range(0, image.shape[0], windowsize_r):
		count = 0
		for c in range(0, image.shape[1], windowsize_c):
			window = image[r:r+windowsize_r,c:c+windowsize_c]
			img_pic = Image.fromarray(window)
			img_path = os.path.join(res_dir, dataset_names[dataset][count]) # The attribute folder in that dataset's folder
			if not os.path.exists(img_path):
				os.mkdir(img_path)
			img_nm = 'img'+str(img_count)+'.jpg'
			img_path = os.path.join(img_path, img_nm) 
			img_pic.save(img_path)
			count+=1
			img_count+=1

'''
Iterate over all generated images in a particular dataset's results folder.

Arguments:
	dataset (str): dataset to be working on
	results_dir (str): the results directory which contain the output of StarGAN
'''
def get_test_imgs(dataset, results_dir):

	for img in os.listdir(results_dir):
		image_file = os.path.join(results_dir, img)
		if os.path.isfile(image_file) and image_file.endswith(".jpg"):
			image_arr = np.array(Image.open(image_file))
			get_single_imgs(dataset, results_dir, image_arr)

if __name__ == '__main__':

	if not os.path.exists(res_for_resnet_dir):
		os.mkdir(res_for_resnet_dir)

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--results_dir', type=str)

	config = parser.parse_args()

	get_test_imgs(config.dataset, config.results_dir)
