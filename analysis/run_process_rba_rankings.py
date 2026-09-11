# run_process_rba_rankings.py
import os
import time
import argparse

def generate_job_script(scheduler, python_script, root_dir, job_name, job_dir, log_dir):
    job_script_path = os.path.join(job_dir, f"{job_name}.sh")

    if scheduler == 'lsf':
        script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{job_name}.out
#BSUB -e {log_dir}/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 02:00
#BSUB -q i2c2_normal

python {python_script} "{root_dir}"
"""
        submit_cmd = f"bsub < {job_script_path}"

    elif scheduler == 'slurm':
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=defq

python {python_script} "{root_dir}"
"""
        submit_cmd = f"sbatch {job_script_path}"

    else:
        raise ValueError("Unsupported scheduler type. Use 'lsf' or 'slurm'.")

    with open(job_script_path, 'w') as f:
        f.write(script)

    os.system(submit_cmd)


def main():
    parser = argparse.ArgumentParser(description="Submit a job to compute global RBA rankings.")
    parser.add_argument('--rootdir', required=True, help='Root directory containing results folders.')
    parser.add_argument('--hpctype', choices=['lsf', 'slurm'], default='slurm', help='Scheduler type: lsf or slurm.')
    parser.add_argument('--script', default='job_process_rba_rankings.py', help='Python script to run.')
    parser.add_argument('--jobdir', default='jobs', help='Directory to store job scripts.')
    parser.add_argument('--logdir', default='logs', help='Directory to store output and error logs.')

    args = parser.parse_args()

    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    root_basename = os.path.basename(os.path.normpath(args.rootdir))

    job_name = f"rba_rankings_{root_basename}_{int(time.time())}"
    generate_job_script(args.hpctype, args.script, args.rootdir, job_name, args.jobdir, args.logdir)
    print(f"Submitted single job for: {args.rootdir}")


if __name__ == "__main__":
    main()