import os

def find_error_runs(log_dir, error_keywords=None):
    if error_keywords is None:
        # Add more as needed
        error_keywords = ["error", "failed", "segmentation fault", "traceback", "exception", "abort", "core dumped"]

    error_runs = []

    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith((".err", ".log", ".out")):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in error_keywords):
                            error_runs.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return error_runs

# Example usage:
log_directory = "./logs"  # Change this to your logs directory
errors = find_error_runs(log_directory)

print("Jobs with errors:")
for err_file in errors:
    print(err_file)
