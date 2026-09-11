import os
import time

ALGORITHMS = [
    "mutual_info",
    "relieff10",
    "relieff100",
    "surf",
    "surfstar",
    "multisurf",
    "multisurfstar",
    "random",
    "swrfstar",
    "swrf",
    "multiswrfstar",
    "multiswrf",
    "multiswrfdbstar",
    "multiswrfdb",
    "murelief10",
    "murelief100"
]

LSF_TEMPLATE = """#!/bin/bash
#BSUB -J {job_name}
#BSUB -o logs/{job_name}.out
#BSUB -e logs/{job_name}.err
#BSUB -n 1
#BSUB -R "rusage[mem=4096]"
#BSUB -W 01:00
#BSUB -q i2c2_normal

python job_rebate_analysis.py --algorithm {algorithm} --input_file "{input_file}"
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq

python job_rebate_analysis.py --algorithm {algorithm} --input_file "{input_file}"
"""

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_dataset_files(root_dir):
    dataset_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "EDM" in os.path.basename(dirpath) or os.path.basename(dirpath) == "3Class_Datasets_Loc_2" or os.path.basename(dirpath) == "9Class_Datasets_Loc_2":
            for filename in filenames:
                if filename.endswith(".txt"):
                    dataset_files.append(os.path.join(dirpath, filename))
    return dataset_files

def generate_hpc_jobs(data_dir, dataset_files, algorithms, job_dir, suffix, hpctype="slurm"):
    ensure_dir(job_dir)
    if hpctype == "lsf":
        TEMPLATE = LSF_TEMPLATE
    elif hpctype == "slurm":
        TEMPLATE = SLURM_TEMPLATE
    else:
        raise Exception(f"Unsupported HPC type: {hpctype}")
    job_paths = []
    for dataset in dataset_files:
        rel_path = os.path.relpath(dataset, data_dir)
        parts = rel_path.split(os.sep) # Split into parts
        last_dirs_and_file = parts[-3:]  # [-3:] = last 2 dirs + file
        safe_path = "_".join(last_dirs_and_file) # Join with underscore
        safe_path = safe_path.replace(".", "")
        base_name = os.path.splitext(safe_path)[0]
        for algo in algorithms:
            job_name = f"{base_name}_{algo}_{suffix}"
            job_path = os.path.join(job_dir, f"{job_name}.sh")
            with open(job_path, "w") as f:
                f.write(TEMPLATE.format(
                    job_name=job_name,
                    algorithm=algo,
                    input_file=dataset,
                ))
            job_paths.append(job_path)
    return job_paths

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, help="Root directory to search for dataset files")
    parser.add_argument("--jobdir", default="jobs", help="Directory to save LSF job scripts")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS, help="Algorithms to run")
    parser.add_argument("--hpctype", choices=["lsf", "slurm"], default="slurm", help="HPC type (lsf or slurm)")
    args = parser.parse_args()

    suffix = time.strftime("%Y%m%d_%H%M%S")
    dataset_files = find_dataset_files(args.datadir)
    job_files = generate_hpc_jobs(args.datadir, dataset_files, args.algorithms, args.jobdir, suffix, hpctype=args.hpctype)
    ensure_dir(os.path.join("", "logs"))

    print(f"Generated {len(job_files)} job scripts in: {args.jobdir}")

    # Automatically submit jobs
    for job_file in job_files:
        if args.hpctype == "lsf":
            os.system(f"bsub < {job_file}")
        elif args.hpctype == "slurm":
            os.system(f"sbatch {job_file}")
        else:
            raise Exception(f"Unsupported HPC type: {args.hpctype}")
        print(f"Submitted job: {job_file}")

    print("All jobs submitted successfully.")

if __name__ == "__main__":
    main()
