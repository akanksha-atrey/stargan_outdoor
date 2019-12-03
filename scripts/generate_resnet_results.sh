#!/bin/bash
#
#SBATCH --job-name=gen_resnet_results
#SBATCH --output=results_resnet/res_%j.txt  # output file
#SBATCH -e results_resnet/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --nodes=7
#SBATCH --ntasks=7
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source venv/bin/activate

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset transient --results_dir './stargan_seasonal/{1,10,10,1e-5,1e-5}/results' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset landmarks --results_dir './stargan_landmarks/{1,15,15,1e-4,1e-4}/results' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset world_cities --results_dir './stargan_wc/results' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset both_sl --results_dir './stargan_both_sl/{1,15,15,1e-4,1e-4}/results_seasonal' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset both_sl --results_dir './stargan_both_sl/{1,15,15,1e-4,1e-4}/results_landmarks' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset both_sc --results_dir './stargan_both_sc/{1,5,5,1e-5,1e-5}/results_seasonal' &

srun -N 1 --ntasks 1 python utils/generate_test_imgs.py --dataset both_sc --results_dir './stargan_both_sc/{1,5,5,1e-5,1e-5}/results_cities' &

wait
 
echo "Done"

hostname
sleep 1
exit
