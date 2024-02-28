import traceback

from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("compile")
  add_common_args(parser)

  parser.add_argument("-t", "--run-tests", action="store_true",
                              help="Run tests on the compiled models.")
  parser.add_argument("--tests-atol", type=float, default=1.e-5,
                              help="The absolute tolerance for float comparisons in tests.")
  parser.add_argument("--fail-on-error", action="store_true",
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
