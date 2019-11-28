#!/bin/bash
#
#SBATCH --job-name=train_sg_wc_hp
#SBATCH --output=stargan_wc/res_%j.txt  # output file
#SBATCH -e stargan_wc/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source venv/bin/activate

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,15,15,1e-3,1e-3}/samples' \
		 --log_dir 'stargan_wc/{1,15,15,1e-3,1e-3}/logs' --model_save_dir 'stargan_wc/{1,15,15,1e-3,1e-3}/models' --result_dir 'stargan_wc/{1,15,15,1e-3,1e-3}/results' \
		 --lambda_cls 1 --lambda_rec 15 --lambda_gp 15 --g_lr 0.001 --d_lr 0.001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,10,10,1e-3,1e-3}/samples' \
		 --log_dir 'stargan_wc/{1,10,10,1e-3,1e-3}/logs' --model_save_dir 'stargan_wc/{1,10,10,1e-3,1e-3}/models' --result_dir 'stargan_wc/{1,10,10,1e-3,1e-3}/results' \
		 --lambda_cls 1 --lambda_rec 10 --lambda_gp 10 --g_lr 0.001 --d_lr 0.001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,5,5,1e-3,1e-3}/samples' \
		 --log_dir 'stargan_wc/{1,5,5,1e-3,1e-3}/logs' --model_save_dir 'stargan_wc/{1,5,5,1e-3,1e-3}/models' --result_dir 'stargan_wc/{1,5,5,1e-3,1e-3}/results' \
		 --lambda_cls 1 --lambda_rec 5 --lambda_gp 5 --g_lr 0.001 --d_lr 0.001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,15,15,1e-4,1e-4}/samples' \
		 --log_dir 'stargan_wc/{1,15,15,1e-4,1e-4}/logs' --model_save_dir 'stargan_wc/{1,15,15,1e-4,1e-4}/models' --result_dir 'stargan_wc/{1,15,15,1e-4,1e-4}/results' \
		 --lambda_cls 1 --lambda_rec 15 --lambda_gp 15 --g_lr 0.0001 --d_lr 0.0001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,5,5,1e-4,1e-4}/samples' \
		 --log_dir 'stargan_wc/{1,5,5,1e-4,1e-4}/logs' --model_save_dir 'stargan_wc/{1,5,5,1e-4,1e-4}/models' --result_dir 'stargan_wc/{1,5,5,1e-4,1e-4}/results' \
		 --lambda_cls 1 --lambda_rec 5 --lambda_gp 5 --g_lr 0.0001 --d_lr 0.0001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,15,15,1e-5,1e-5}/samples' \
		 --log_dir 'stargan_wc/{1,15,15,1e-5,1e-5}/logs' --model_save_dir 'stargan_wc/{1,15,15,1e-5,1e-5}/models' --result_dir 'stargan_wc/{1,15,15,1e-5,1e-5}/results' \
		 --lambda_cls 1 --lambda_rec 15 --lambda_gp 15 --g_lr 0.00001 --d_lr 0.00001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,10,10,1e-5,1e-5}/samples' \
		 --log_dir 'stargan_wc/{1,10,10,1e-5,1e-5}/logs' --model_save_dir 'stargan_wc/{1,10,10,1e-5,1e-5}/models' --result_dir 'stargan_wc/{1,10,10,1e-5,1e-5}/results' \
		 --lambda_cls 1 --lambda_rec 10 --lambda_gp 10 --g_lr 0.00001 --d_lr 0.00001 &

srun -N 1 --ntasks 1 python code/main.py --mode train --dataset RaFD --c_dim 5 --rafd_image_dir data/world_cities_outdoor/train --sample_dir 'stargan_wc/{1,5,5,1e-5,1e-5}/samples' \
		 --log_dir 'stargan_wc/{1,5,5,1e-5,1e-5}/logs' --model_save_dir 'stargan_wc/{1,5,5,1e-5,1e-5}/models' --result_dir 'stargan_wc/{1,5,5,1e-5,1e-5}/results' \
		 --lambda_cls 1 --lambda_rec 5 --lambda_gp 5 --g_lr 0.00001 --d_lr 0.00001 &

wait
 
echo "Done"

hostname
sleep 1
exit
