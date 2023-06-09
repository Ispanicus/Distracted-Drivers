import concurrent.futures
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_WORKERS = 1
HYPERPARAMETERS_FILE = "hyperparameters_3.json"
classify_path = str((Path(__file__).parent / "classifiers.py").absolute())
venv_python_path = str((Path(__file__).parents[2] / "venv/Scripts/python").absolute())
hyper_params_path = str((Path(__file__).parent / HYPERPARAMETERS_FILE).absolute())


def task(**kwargs):
    cmd = [classify_path]
    for key, value in kwargs.items():
        cmd.append(key)
        cmd.append(str(value))
    result = subprocess.Popen([venv_python_path] + cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()
    print(stdout)
    if stderr:
        print(stderr.decode("utf-8"))
    return result


def main():
    results = dict()

    with open(hyper_params_path, "r") as f:
        hyperparameters = json.load(f)

    for run_slug, kwargs in hyperparameters.items():
        print(f"Starting {run_slug}")
        result = task(**kwargs)
        results[run_slug] = {run_slug: {"results": result, "kwargs": kwargs}}
        print(f"Finished {run_slug}")


def main2():
    results = dict()

    with open(hyper_params_path, "r") as f:
        hyperparameters = json.load(f)

    print("starting ThreadPoolExecutor")
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_dict = {
            executor.submit(task, **kwargs): {run_slug: kwargs}
            for run_slug, kwargs in hyperparameters.items()
        }
        for future in concurrent.futures.as_completed(future_dict):
            run_dict = future_dict[future]
            run_slug = list(run_dict.keys())[0]
            kwargs = run_dict[run_slug]
            try:
                data = future.result()
            except Exception as exc:
                print(f"{run_slug} generated an exceptions: {exc} ")
            else:
                results[run_slug] = {run_slug: {"results": data, "kwargs": kwargs}}
                if data.returncode != 0:
                    _, stderr = data.communicate()
                    errors.append(stderr.decode("utf-8"))
                    print(stderr.decode("utf-8"))
    print("Finished ThreadPoolExecutor")
    if errors:
        print("Errors detected!")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()
