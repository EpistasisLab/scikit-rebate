# run_global_rba_rankings.py
import os
import argparse
import time

def generate_job_script(scheduler, python_script, root_dir, job_name, job_dir, log_dir, include_subdirs=None):
    job_script_path = os.path.join(job_dir, f"{job_name}.sh")
    include_arg = ''
    if include_subdirs:
        include_arg = ' '.join(include_subdirs)
        include_arg = f"--include {include_arg}"

    if scheduler == 'lsf':
        script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{job_name}.out
#BSUB -e {log_dir}/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 02:00
#BSUB -q i2c2_normal

python {python_script} "{root_dir}" {include_arg}
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

python {python_script} "{root_dir}" {include_arg}
"""
        submit_cmd = f"sbatch {job_script_path}"

    else:
        raise ValueError("Unsupported scheduler type. Use 'lsf' or 'slurm'.")

    with open(job_script_path, 'w') as f:
        f.write(script)

    os.system(submit_cmd)
    print(f"[INFO] Submitted job: {job_name}")


def main():
    parser = argparse.ArgumentParser(description="Submit global RBA rankings job(s).")
    parser.add_argument('--rootdir', required=True, help='Root directory containing dataset subdirectories.')
    parser.add_argument('--hpctype', choices=['lsf', 'slurm'], default='slurm', help='Scheduler type.')
    parser.add_argument('--script', default='job_global_rba_rankings.py', help='Python script to run.')
    parser.add_argument('--jobdir', default='jobs', help='Directory to store job scripts.')
    parser.add_argument('--logdir', default='logs', help='Directory to store logs.')
    parser.add_argument('--include', nargs='+', default=None, help='Optional list of subdirectories to include (short names).')
    args = parser.parse_args()

    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    job_name = f"global_rba_{int(time.time())}"
    generate_job_script(args.hpctype, args.script, args.rootdir, job_name, args.jobdir, args.logdir, args.include)


if __name__ == "__main__":
    main()