import subprocess
import os


def run_commands(commands):
    dirname = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    print(f"Running commands in {dirname}")
    for command in commands:
        command[1] = os.path.join(dirname, command[1])
        print(f"Running command: {' '.join(command)}")
        command = " ".join(command)
        subprocess.run(command, check=True, shell=True)
