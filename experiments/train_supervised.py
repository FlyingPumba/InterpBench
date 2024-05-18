#!/usr/bin/env python3
import subprocess
import sys
from itertools import product
from math import ceil
from pathlib import Path
from typing import List
from utils import *

import numpy as np

from circuits_benchmark.utils.get_cases import get_cases

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands():
    case_instances = get_cases(indices=None)
    cases = []