from circuits_benchmark.utils.get_cases import get_names_of_working_cases
import os
from tqdm import tqdm
import time
import sys
import subprocess

working_cases = sorted(get_names_of_working_cases())
print(working_cases)

ioi = [case for case in working_cases if "ioi" in case]
for ioi_case in ioi:
    working_cases.remove(ioi_case)
# cases = [11, 3]
for task in tqdm(working_cases):
    commands = []
    for algorithm in ["acdc", "eap", "integrated_grad"]:
        # os.system('cls' if os.name == 'nt' else 'clear')
        command = f"python run_locally_circuit_discovery.py --case {task} --clean --algorithm {algorithm}"
        commands.append(command)
    
    # run all commands and wait for them to finish
    processes = []
    for command in commands:
        processes.append(subprocess.Popen(command, shell=True))
    for process in processes:
        process.wait()

