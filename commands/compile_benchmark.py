import traceback

import numpy as np
import torch
import torch as t

from benchmark.benchmark_case import BenchmarkCase
from compression.residual_stream import compress, setup_compression_training_args_for_parser
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.compiling import TracrOutput
from tracr.transformer.encoder import CategoricalEncoder
from utils.get_cases import get_cases
from utils.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  compile_parser = subparsers.add_parser("compile")
  compile_parser.add_argument("-i", "--indices", type=str, default=None,
                              help="A list of comma separated indices of the cases to compile. "
                                   "If not specified, all cases will be compiled.")
  compile_parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                              help="The device to use for compression.")
  compile_parser.add_argument("-f", "--force", action="store_true",
                              help="Force compilation of cases, even if they have already been compiled.")
  compile_parser.add_argument("-t", "--run-tests", action="store_true",
                              help="Run tests on the compiled models.")
  compile_parser.add_argument("--tests-atol", type=float, default=1.e-5,
                              help="The absolute tolerance for float comparisons in tests.")
  compile_parser.add_argument("--fail-on-error", action="store_true",
                              help="Fail on error and stop compilation.")

  setup_compression_training_args_for_parser(compile_parser)


def compile_all(args):
  for case in get_cases(args):
    print(f"\nCompiling {case}")
    try:
      tracr_output = build_tracr_model(case, args.force)

      if args.run_tests:
        run_case_tests_on_tracr_model(case, tracr_output.model, args.tests_atol)

      tl_model = build_transformer_lens_model(case,
                                              force=args.force,
                                              tracr_output=tracr_output,
                                              device=args.device)

      if args.run_tests:
        run_case_tests_on_tl_model(case, tl_model, args.tests_atol)

      if args.compress_residual is not None:
        compress(case, tl_model, args.compress_residual, args)

    except Exception as e:
      print(f" >>> Failed to compile {case}:")
      traceback.print_exc()

      if args.fail_on_error:
        raise e
      else:
        continue


def build_tracr_model(case: BenchmarkCase, force: bool = False) -> TracrOutput:
  """Compiles a single case to a tracr model."""

  # if tracr model and output have already compiled, we just load and return them
  if not force:
    tracr_model = case.load_tracr_model()
    tracr_graph = case.load_tracr_graph()
    if tracr_model is not None and tracr_graph is not None:
      return TracrOutput(tracr_model, tracr_graph)

  program = case.get_program()
  max_seq_len = case.get_max_seq_len()
  vocab = case.get_vocab()

  tracr_output = compiling.compile_rasp_to_model(
    program,
    vocab=vocab,
    max_seq_len=max_seq_len,
    compiler_bos="BOS",
  )

  # write tracr model and graph to disk
  case.dump_tracr_model(tracr_output.model)
  case.dump_tracr_graph(tracr_output.graph)

  return tracr_output


def build_transformer_lens_model(case: BenchmarkCase,
                                 force: bool = False,
                                 tracr_output: TracrOutput = None,
                                 device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
                                 ) -> HookedTracrTransformer:
  """Compiles a tracr model to transformer lens."""
  if not force:
    tl_model = case.load_tl_model()
    if tl_model is not None:
      tl_model.to(device)
      return tl_model

  tracr_model = None
  if tracr_output is not None:
    tracr_model = tracr_output.model

  if tracr_model is None:
    tracr_output = build_tracr_model(case, force)
    tracr_model = tracr_output.model

  tl_model = HookedTracrTransformer(tracr_model, device=device)

  case.dump_tl_model(tl_model)

  return tl_model


def run_case_tests_on_tracr_model(case: BenchmarkCase,
                                  tracr_model: AssembledTransformerModel,
                                  atol: float = 1.e-5):
  dataset = case.get_clean_data()
  inputs = dataset[BenchmarkCase.DATASET_INPUT_FIELD]
  expected_outputs = dataset[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD]

  is_categorical = isinstance(tracr_model.output_encoder, CategoricalEncoder)

  for i in range(len(inputs)):
    input = inputs[i]
    expected_output = expected_outputs[i]
    decoded_output = tracr_model.apply(input).decoded

    if is_categorical:
      correct = decoded_output == expected_output
    else:
      # then output is numerical and we need to convert it to floats, because they are stored as strings in the dataset
      expected_output = [float(x) for x in expected_output[1:]]
      decoded_output = [float(x) for x in decoded_output[1:]]
      correct = np.allclose(expected_output, decoded_output, atol=atol)

    if not correct:
      raise ValueError(f"Failed test for {case} on tracr model."
                       f"\n >>> Input: {input}"
                       f"\n >>> Expected: {expected_output}"
                       f"\n >>> Got: {decoded_output}")


def run_case_tests_on_tl_model(case: BenchmarkCase,
                               tl_model: HookedTracrTransformer,
                               atol: float = 1.e-5):
  dataset = case.get_clean_data()
  inputs = dataset[BenchmarkCase.DATASET_INPUT_FIELD]
  expected_outputs = dataset[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD]
  decoded_outputs = tl_model(inputs, return_type="decoded")

  for i in range(len(expected_outputs)):
    input = inputs[i]
    expected_output = expected_outputs[i]
    decoded_output = decoded_outputs[i]

    if tl_model.is_categorical():
      correct = decoded_output == expected_output
    else:
      # then output is numerical and we need to convert it to floats, because they are stored as strings in the dataset
      expected_output = [float(x) for x in expected_output[1:]]
      decoded_output = [float(x) for x in decoded_output[1:]]
      correct = np.allclose(decoded_output, expected_output, atol=atol)

    if not correct:
      raise ValueError(f"Failed test for {case} on tl model."
                       f"\n >>> Input: {input}"
                       f"\n >>> Expected: {expected_output}"
                       f"\n >>> Got: {decoded_output}")