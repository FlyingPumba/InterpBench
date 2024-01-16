#! /usr/bin/env python3
import os
import glob
import importlib
from tracr.compiler import compiling
from benchmark.defaults import default_max_seq_len, default_bos, default_vocab


if __name__ == "__main__":
  # list all cases in the benchmark directory
  benchmark_dir = "benchmark"
  files = sorted(glob.glob(os.path.join(benchmark_dir, "case-*", "rasp.py")))

  for file_path in files:
    try:
      # evaluate the "get_program()" method in the file and compile it.
      print(f"\nCompiling {file_path}")
      
      # load modulle for the file
      module_name = file_path.replace("/", ".")[:-3]
      module = importlib.import_module(module_name)

      # evaluate "get_program()" method dinamically
      get_program_fn = getattr(module, 'get_program')
      program = get_program_fn()
    
      model = compiling.compile_rasp_to_model(
          program,
          vocab=default_vocab,
          max_seq_len=default_max_seq_len,
          compiler_bos=default_bos,
      )
    except Exception as e:
      print(f" >>> Failed to compile {file_path}")
      print(e)
      continue