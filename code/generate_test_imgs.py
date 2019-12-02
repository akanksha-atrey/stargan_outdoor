import numpy as np
from PIL import Image
import os
import argparse
 

# All image files with consolidated results for each dataset, must be in './results/landmarks', './results/world_cities' and './results/transient'
# Once this script is run, the single images are stored as './results_resnet/landmarks/landmark_name/img_name'
# So the results_resnet call be fed directly to the resnet
# This script has to be used as : generate_test_imgs.py --dataset landmarks OR generate_test_imgs.py --dataset world_cities OR generate_test_imgs.py --dataset transient

landmarks = ['original', 'EdinburghCastle', 'EiffelTower', 'GoldenGateBridge', 'GrandCanyon', 'Masada', 'MountRainier', 'NiagaraFalls']
world_cities = ['original', 'Amsterdam','Athens','Beijing','Paris', 'NewYork']
transient = ['original', 'autumn','clouds','dawndusk','daylight','fog','night','rain','snow','spring','storm','summer','sunny','sunrisesunset', 'winterfrom PIL import Image']
dataset_names = {'landmarks': landmarks, 'world_cities': world_cities, 'transient':transient}
results_dir = './results'
res_for_resnet_dir = './results_resnet'
img_count = 0


# Get single images for each consolidated image and saved in their respective class folder
def get_single_imgs(dataset, image):
	# Define the window size
	windowsize_r = 128
	windowsize_c = 128
	
	global img_count
	global dataset_names
	res_dir = os.path.join(res_for_resnet_dir, dataset)
	if not os.path.exists(res_dir):
		os.mkdir(res_dir)
	# Crop out the window and calculate the histogram
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


# Iterate over all generated images in a particular dataset's results folder
def get_test_imgs(dataset):

	global results_dir
	dataset_dir = os.path.join(results_dir, dataset)
	for img in os.listdir(dataset_dir):
		image_file = os.path.join(dataset_dir, img)
		image_arr = np.array(Image.open(image_file))
		get_single_imgs(dataset, image_arr)

if __name__ == '__main__':

	if not os.path.exists(res_for_resnet_dir):
		os.mkdir(res_for_resnet_dir)
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str)
	dataset = parser.parse_args().dataset
	get_test_imgs(dataset)
