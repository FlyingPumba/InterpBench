import traceback

import torch

from utils.get_cases import get_cases


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


def compile_all(args):
  for case in get_cases(args):
    print(f"\nCompiling {case}")
    try:
      tracr_output = case.build_tracr_model()

      if args.run_tests:
        case.run_case_tests_on_tracr_model(tracr_model=tracr_output.model, atol=args.tests_atol)

      tl_model = case.build_transformer_lens_model(tracr_model=tracr_output.model, device=args.device)

      if args.run_tests:
        case.run_case_tests_on_tl_model(tl_model=tl_model, atol=args.tests_atol)

    except Exception as e:
      print(f" >>> Failed to compile {case}:")
      traceback.print_exc()

      if args.fail_on_error:
        raise e
      else:
        continue
