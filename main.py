#! /usr/bin/env python3
import os
import glob
import importlib
import jax
import logging
import traceback
from tracr.compiler import compiling
from benchmark.defaults import default_max_seq_len, default_bos, default_vocab

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

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

      max_seq_len = default_max_seq_len
      if hasattr(module, 'get_max_seq_len'):
          get_max_seq_len_fn = getattr(module, 'get_max_seq_len')
          max_seq_len = get_max_seq_len_fn()

      vocab = default_vocab
      if hasattr(module, 'get_vocab'):
          get_vocab_fn = getattr(module, 'get_vocab')
          vocab = get_vocab_fn()
    
      model = compiling.compile_rasp_to_model(
          program,
          vocab=vocab,
          max_seq_len=max_seq_len,
          compiler_bos=default_bos,
      )
    except Exception as e:
      print(f" >>> Failed to compile {file_path}:")
      traceback.print_exc()
      continue