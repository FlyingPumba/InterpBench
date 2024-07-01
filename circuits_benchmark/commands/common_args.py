import torch as t

from circuits_benchmark.utils.project_paths import get_default_output_dir


def add_common_args(parser):
  parser.add_argument("-i", "--indices", type=str, default=None,
                      help="A list of comma separated indices of the cases to run against. "
                           "If not specified, all cases will be run.")
  parser.add_argument("-o", "--output-dir", type=str, default=get_default_output_dir(),
                      help="The directory to save the results to.")
  parser.add_argument("-d", "--device", type=str, default="cuda" if t.cuda.is_available() else "cpu",
                      help="The device to use for experiments.")
  parser.add_argument('--seed', type=int, default=1234,
                      help='The seed to use for experiments.')


def add_evaluation_common_ags(parser):
  parser.add_argument(
    "--siit-weights",
    type=str,
    default=None,
    help="The weights for the SIIT model that will be evaluated.",
  )
  parser.add_argument(
    "--tracr",
    action="store_true",
    help="Evaluate Tracr-generated model for the selected case."
  )
  parser.add_argument(
    "--natural",
    action="store_true",
    help="Evaluate 'naturally' trained model. "
         "This assumes that the model is already trained and stored in wandb or "
         "<output_dir>/ll_models/<case_name>/ll_model_natural.pth (run train iit for this)",
  )
  parser.add_argument(
    "--interp-bench",
    action="store_true",
    help="Evaluate the InterpBench model for the selected case."
  )
  parser.add_argument(
    "--load-from-wandb",
    action="store_true",
    help="Load the model to evaluate from wandb."
  )