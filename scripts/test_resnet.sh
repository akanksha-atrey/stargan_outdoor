#!/bin/bash
#
#SBATCH --job-name=test_resnet
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

# srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type real &

# srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type fake &

# srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset landmarks --test_data_type real &

# srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset landmarks --test_data_type fake &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset world_cities --test_data_type real &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset world_cities --test_data_type fake &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type fake --testset both_sl/landmarks/seasonal &

# srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset landmarks --test_data_type fake --testset both_sl/landmarks/landmarks &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type fake --testset both_sl/seasonal/seasonal &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type fake --testset both_sc/cities/seasonal &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset world_cities --test_data_type fake --testset both_sc/cities/cities &

srun -N 1 --ntasks 1 python code/resnet18.py --mode 'test' --dataset transient --test_data_type fake --testset both_sc/seasonal/seasonal &

wait
 
echo "Done"

hostname
sleep 1
exit
