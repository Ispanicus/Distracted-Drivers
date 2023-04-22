from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json
from distracted.data_util import Hyperparameters
import subprocess
import os


MAX_WORKERS=8

def task(**kwargs):
    cmd = ["python", "../classifiers.py"]
    key_to_arg = {"batch_size": "--batch-size", "model_name": "--model-name", "adapters": "--adapters", "top_lr": "--top-lr", "top_decay": "--top-decay", "body_lr": "--body-lr", "body_decay": "--body-decay", "gamma": "--gamma", "epochs": "--epochs", }
    for key, value in kwargs.items():
        cmd.append(key_to_arg[key])
        cmd.append(str(value))
    result = subprocess.run(cmd, capture_output=True)
    return result

def main():
    results = dict()

    with open("hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    # Validation
    for run_slug, hyperparameter in hyperparameters.items():
        hyperparameter = Hyperparameters(**hyperparameter)
        hyperparameters[run_slug] = hyperparameter.dict()

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
            results[run_slug] = {run_slug:{"results":data, "kwargs":kwargs}}


if __name__ == "__main__":
    main()