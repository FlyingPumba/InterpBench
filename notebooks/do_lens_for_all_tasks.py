import subprocess
from circuits_benchmark.utils.get_cases import get_names_of_working_cases, get_names_of_categorical_cases


cases = get_names_of_working_cases()
cases = [case for case in cases if "ioi" not in case]
ll_script = "python logit_lens_script.py --task"
# tl_script = "python tuned_lens_script.py --task"
max_subprocesses = 10
processes = []
for case in cases:
    ll_script_to_run = f"{ll_script} {case}"
    # tl_script_to_run = f"{tl_script} {case}"

    ll_process = subprocess.Popen(ll_script_to_run, shell=True)
    # tl_process = subprocess.Popen(tl_script_to_run, shell=True)
    processes.append(ll_process)
    # processes.append(tl_process)

    if len(processes) >= max_subprocesses:
        while True:
            for process in processes:
                # check if the process is still running
                if process.poll() is not None:
                    # remove the finished process
                    processes.remove(process)
                    print(f"Process {process} finished")
                    break
                # else:
                    # print(f"\rProcess {process} still running")
            if len(processes) < max_subprocesses:
                break
