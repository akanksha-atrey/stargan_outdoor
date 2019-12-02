#!/bin/bash
#
#SBATCH --job-name=test_stargan_wc
#SBATCH --output=stargan_wc/res_%j.txt         # output file
#SBATCH -e stargan_wc/res_%j.err               # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source venv/bin/activate

python code/main.py --mode test --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/test \
					--sample_dir stargan_wc/samples \
					--log_dir stargan_wc/logs \
					--model_save_dir stargan_wc/models \
					--result_dir stargan_wc/results

echo "Done"

hostname
sleep 1
exit
