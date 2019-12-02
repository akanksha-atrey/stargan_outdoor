#!/bin/bash
#
#SBATCH --job-name=test_stargan_sc
#SBATCH --output=stargan_both_sc/res_%j.txt  # output file
#SBATCH -e stargan_both_sc/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source venv/bin/activate

python code/main.py --mode test --dataset Both --c_dim 5 --c2_dim 14 \
					--celeba_image_dir data/world_cities_outdoor/test --rafd_image_dir data/transient_attributes/test \
					--sample_dir 'stargan_both_sc/{1,5,5,1e-5,1e-5}/samples' \
					--log_dir 'stargan_both_sc/{1,5,5,1e-5,1e-5}/logs' \
					--model_save_dir 'stargan_both_sc/{1,5,5,1e-5,1e-5}/models' \
					--result_dir 'stargan_both_sc/{1,5,5,1e-5,1e-5}/results_cities'

echo "Done"

hostname
sleep 1
exit
