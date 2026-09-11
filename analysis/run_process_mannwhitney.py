# run_process_mannwhitney.py
import os
import time
import argparse

def generate_job_script(scheduler, python_script, dir_path, job_name, job_dir, log_dir, column, exclude):
    job_script_path = os.path.join(job_dir, f"{job_name}.sh")
    column_flag = f"--column {column}" if column else ""
    exclude_flag = f"--exclude {' '.join(exclude)}" if exclude else ""

    if scheduler == 'lsf':
        script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{job_name}.out
#BSUB -e {log_dir}/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 03:00
#BSUB -q i2c2_normal

python {python_script} "{dir_path}" {column_flag} {exclude_flag}
"""
        submit_cmd = f"bsub < {job_script_path}"

    elif scheduler == 'slurm':
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --partition=defq

python {python_script} "{dir_path}" {column_flag} {exclude_flag}
"""
        submit_cmd = f"sbatch {job_script_path}"
    else:
        raise ValueError("Unsupported scheduler type. Use 'lsf' or 'slurm'.")

    with open(job_script_path, 'w') as f:
        f.write(script)

    os.system(submit_cmd)

def main():
    parser = argparse.ArgumentParser(description="Generate and submit Mann-Whitney U test jobs.")
    parser.add_argument('--basedir', required=True, help='Base directory to search for rankings_list.csv.')
    parser.add_argument('--hpctype', choices=['lsf', 'slurm'], default='slurm', help='Scheduler type: lsf or slurm.')
    parser.add_argument('--script', default='job_process_mannwhitney.py', help='Python script to run.')
    parser.add_argument('--jobdir', default='jobs', help='Directory to store job scripts.')
    parser.add_argument('--logdir', default='logs', help='Directory to store output and error logs.')
    parser.add_argument('--column', choices=['rank', 'feature_importance'], default='rank', help='Column to use for tests.')
    parser.add_argument('--exclude', nargs='*', default=[], help='List of patterns to exclude RBAs.')

    args = parser.parse_args()

    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    job_count = 0
    dir_path = args.basedir  # only look directly inside basedir
    if 'rankings_list.csv' in os.listdir(dir_path):
        job_name = f"mannwhitney_job_{job_count}_{int(time.time())}"
        generate_job_script(args.hpctype, args.script, dir_path, job_name, args.jobdir, args.logdir, args.column, args.exclude)
        job_count += 1

if __name__ == "__main__":
    main()