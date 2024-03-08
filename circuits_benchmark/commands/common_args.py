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