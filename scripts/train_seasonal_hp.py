from sbatch import sbatch
from pathlib import Path
from datetime import datetime
import math
from itertools import product
import numpy as np
import time
import subprocess
import os

def get_current_jobs():
     process = subprocess.Popen('squeue -u aatrey | wc -l', stdout=subprocess.PIPE, shell=True)
     output, error = process.communicate()
     return int(output) - 1

def main():
    log_dir = 'stargan_seasonal'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d.%H%M%S')

    lambda_cls = [1,5,10,15]
    lambda_rec = [1,5,10,15]
    lambda_gp = [1,5,10,15]
    g_lr = [1e-3, 1e-4, 1e-5]
    d_lr = [1e-3, 1e-4, 1e-5]

    configurations = list(product(set(lambda_cls), set(lambda_rec), set(lambda_gp), set(g_lr), set(d_lr)))
#    configurations = [[60, 0]]
    num_configurations = len(configurations)
    print('num_configurations are {}'.format(num_configurations))
    max_number_jobs = 400
    total_num_short_jobs = (12 + 40 + 40)
    for i, config in enumerate(configurations):
        print(config)
        while True:
            try:
                if get_current_jobs() > max_number_jobs:
                    time.sleep(120)
                    continue
                args = ['/venv/bin/python', 'code/main.py']
                args.extend(['--mode', 'train'])
                args.extend(['--dataset', 'RaFD'])
                args.extend(['--c_dim', str(14)])
                args.extend(['--rafd_image_dir', 'data/transient_attributes/train'])
                args.extend(['--sample_dir', 'stargan_seasonal/{}/samples'.format(str(config))])
                args.extend(['--log_dir', 'stargan_seasonal/{}/logs'.format(str(config))])
                args.extend(['--model_save_dir', 'stargan_seasonal/{}/models'.format(str(config))])
                args.extend(['--result_dir', 'stargan_seasonal/{}/results'.format(str(config))])
                args.extend(['--lambda_cls', str(config[0])])
                args.extend(['--lambda_rec', str(config[1])])
                args.extend(['--lambda_gp', str(config[2])])
                args.extend(['--g_lr', str(config[3])])
                args.extend(['--d_lr', str(config[4])])

                name = 'res_{}_{}'.format(timestamp, str(i))
                job_name = 'train_sg_seasonal{}'.format(timestamp)
                stdout = log_dir / str(config) / '{}_o.txt'.format(name)
                stderr = log_dir / str(config) / '{}_e.txt'.format(name)
                command = ' '.join(args)
                sbatch(command, job_name=job_name, stdout=stdout, stderr=stderr, mem='32G', cpus_per_task=8, ntasks_per_node=12, mem_per_cpu=20000, queue='titanx-long', gres='gpu:1')
            except:
                continue
            break

if __name__ == '__main__':
    main()
