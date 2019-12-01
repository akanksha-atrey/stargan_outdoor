import argparse
import glob

def hp_search(path, num_iter):

	output_files = glob.glob("{}/*.txt".format(path))
	hp_search_file = open("{}/hp_search.txt".format(path), "w") 

	for file in output_files:
		with open(file) as f:
			lines = f.readlines()

		for i,line in enumerate(lines):
			if "Saved real and fake images" in line and "/samples/{}-images.jpg".format(num_iter) in line:
				hp = line.split('/')[1]
				loss_output = lines[i-1]

				print('{} : {}'.format(hp, loss_output))
				hp_search_file.write('{} : {} \n'.format(hp, loss_output))

	hp_search_file.close()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--output_path', default='./stargan_seasonal', help='Path which contains the loss output text files.', type=str)
	parser.add_argument('--num_iter', default='200000', help='Number of iterations the model was trained for.', type=str)
	config = parser.parse_args()

	hp_search(config.output_path, config.num_iter)