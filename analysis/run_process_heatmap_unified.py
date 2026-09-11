import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Generate and submit one job for unified heatmap.")
    parser.add_argument('--basedir', required=True, help='Directory containing multiple Results folders.')
    parser.add_argument('--hpctype', choices=['lsf', 'slurm'], default='slurm', help='Scheduler type.')
    parser.add_argument('--script', default='job_process_heatmap_unified.py', help='Python script to run.')
    parser.add_argument('--jobdir', default='jobs', help='Directory to store job scripts.')
    parser.add_argument('--logdir', default='logs', help='Directory to store output and error logs.')
    parser.add_argument('--prefix', default='', help='Prefix for unified PDF filename.')
    args = parser.parse_args()

    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # We’ll pass the whole basedir once to the script
    job_name = f"unified_job_{int(time.time())}"
    job_script_path = os.path.join(args.jobdir, f"{job_name}.sh")
    prefix_flag = f"--prefix {args.prefix}" if args.prefix else ""

    if args.hpctype == 'lsf':
        script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {args.logdir}/{job_name}.out
#BSUB -e {args.logdir}/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 02:00
#BSUB -q i2c2_normal

python {args.script} "{args.basedir}" {prefix_flag}
"""
        submit_cmd = f"bsub < {job_script_path}"
    elif args.hpctype == 'slurm':
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={args.logdir}/{job_name}.out
#SBATCH --error={args.logdir}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=defq

python {args.script} "{args.basedir}" {prefix_flag}
"""
        submit_cmd = f"sbatch {job_script_path}"
    else:
        raise ValueError("Unsupported scheduler type. Use 'lsf' or 'slurm'.")

    with open(job_script_path, 'w') as f:
        f.write(script)

    os.system(submit_cmd)

if __name__ == "__main__":
    main()