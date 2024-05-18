#!/usr/bin/env python3
import subprocess
import sys
from itertools import product
from math import ceil
from pathlib import Path
from typing import List
from utils import *
import numpy as np

def build_commands():
  # command = "python main.py train iit -i 3 --wandb_entity cybershiptrooper".split()
  command = "python main.py run sp --loss-type l2 -i 3 --using-wandb --edgewise --torch-num-threads 4 --device cpu --lambda-reg 1 --epochs 3000".split()
  command[0] = "python"
  return [command]


if __name__ == "__main__":
  print_commands(build_commands)
  launch_kubernetes_jobs(build_commands, cpu = 1, gpu = 1, memory = "24Gi", priority = "high-batch")
