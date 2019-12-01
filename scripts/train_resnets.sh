#!/bin/bash
#
#SBATCH --job-name=train_resnets
#SBATCH --output=resnets/res_%j.txt  # output file
#SBATCH -e resnets/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source venv/bin/activate

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'train' --n_class 5 --dataset 'world_cities' --pretrained False --lr 0.004 --epochs 100 &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'train' --n_class 14 --dataset 'transient' --pretrained False --lr 0.004 --epochs 100 &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'train' --n_class 7 --dataset 'landmarks' --pretrained True --lr 0.004 --epochs 100 &

wait

echo "Done"

hostname
sleep 1
exit
