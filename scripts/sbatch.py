from subprocess import Popen, PIPE
from pathlib import Path
from datetime import datetime


def sbatch(
        command,
        job_name,
        queue='m40-long',
        cpus_per_task=21,
        stdout=None,
        stderr=None,
        copy_env=True,
        email='',
        notification='FAIL',   # BEGIN, END, FAIL, ALL
        workdir=None,
        time=None,   # <mm>, <mm:ss>, <hh:mm:ss>, <days-hh:mm:ss>
        mem=None,
        signal=None,   # [B:]<sig_num>[@<sig_time>]
        depend=None,   # state:job (state: after/afterany/afternotok/afterok*)
        gres='gpu:1',
        exclude=None,   # node[000-003,007,010-013,017]
):

    args = ['sbatch',
            '--job-name={}'.format(job_name),
            '-p', queue]

    if cpus_per_task is not None and cpus_per_task > 0:
        args.extend(['-c', str(cpus_per_task)])

    if stdout is not None:
        args.extend(['-o', str(stdout)])

    if stderr is not None:
        args.extend(['-e', str(stderr)])

    if copy_env:
        args.append('--export=ALL')

    if email is not None:
        args.append('--mail-user={}'.format(email))

    if notification is not None:
        args.append('--mail-type={}'.format(notification))

    if workdir is None:
        args.append('--workdir={}'.format(Path.cwd()))
    else:
        args.append('--workdir={}'.format(workdir))

    if time is not None:
        args.append('--time={}'.format(time))

    if mem is not None:
        args.append('--mem={}'.format(mem))

    if signal is not None:
        args.append('--signal={}'.format(signal))

    if depend is not None:
        if not isinstance(depend, str):
            depend = ','.join(depend)
        args.append('--depend={}'.format(depend))

    if gres is not None:
        args.append('--gres={}'.format(gres))

    if exclude is not None:
        args.append('--exclude={}'.format(exclude))

    script = '#!/bin/bash\nexec {}\n'.format(command).encode('ascii')
    out = Popen(args, stdin=PIPE, stdout=PIPE).communicate(script)[0]
    job_id = out.decode('ascii').split()[3]
    return job_id


def main():
    log_dir = Path.home() / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    stdout = log_dir / '{}_o.txt'.format(timestamp)
    stderr = log_dir / '{}_e.txt'.format(timestamp)
    sbatch('ls -l', job_name='test', stdout=stdout, stderr=stderr)


if __name__ == '__main__':
    main()