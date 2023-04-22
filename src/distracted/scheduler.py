from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json
import subprocess
import os


MAX_WORKERS=8

base_path = os.path.dirname(__file__)
relative_path = "../classifiers.py"
full_path = os.path.join(base_path, relative_path)

def task(**kwargs):
    cmd = ["python", full_path]
    for key, value in kwargs.items():
        cmd.append(key)
        cmd.append(str(value))
    result = subprocess.run(cmd, capture_output=True)
    return result

def main():
    results = dict()

    with open("hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    print("starting ThreadPoolExecutor")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_dict = {executor.submit(task, **kwargs): {run_slug:kwargs} for run_slug, kwargs in hyperparameters.items()}
        for future in concurrent.futures.as_completed(future_dict):
            run_dict = future_dict[future]
            run_slug = list(run_dict.keys())[0]
            kwargs = run_dict[run_slug]
            try:
                data = future.result()
            except Exception as exc:
                print(f"{run_slug} generated an exceptions: {exc} ")
            else:
                results[run_slug] = {run_slug:{"results":data, "kwargs":kwargs}}
    print("Finished ThreadPoolExecutor")

if __name__ == "__main__":
    main()