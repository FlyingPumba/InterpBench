
def setup_args_parser(subparsers):
  acdc_parser = subparsers.add_parser("acdc")
  acdc_parser.add_argument("-i", "--indices", type=str, default=None,
                          help="A list of comma separated indices of the cases to run against. "
                               "If not specified, all cases will be run.")


def run_acdc(file_path, args):
  pass