import subprocess
from circuits_benchmark.utils.get_cases import get_names_of_working_cases
import pandas as pd
csv = pd.read_csv('interp_results/siit_vs_tracr.csv')
remove_cases = [str(t) for t in csv['task_name'].tolist()]
remove_cases += ['ioi', 'ioi_next_token']
cases = get_names_of_working_cases()
cases = [case for case in cases if case not in remove_cases]
print(cases, remove_cases, sep='\n')
# run all cases parallel with max 4 processes
processes = []
for case in cases:
    processes.append(subprocess.Popen(['python', 'siit_vs_tracr.py', '--task', case]))
    if len(processes) >= 4:
        for p in processes:
            p.wait()
            processes.remove(p)
            break # only wait for the first process to finish
    