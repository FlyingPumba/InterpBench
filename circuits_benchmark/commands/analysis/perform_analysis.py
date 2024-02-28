from circuits_benchmark.commands.analysis import compression_matrix


def setup_args_parser(subparsers):
  analysis_parser = subparsers.add_parser("analysis")
  analysis_subparsers = analysis_parser.add_subparsers(dest="type")
  analysis_subparsers.required = True

  # Setup arguments for each analysis
  compression_matrix.setup_args_parser(analysis_subparsers)


def run(args):
  print(f"\nPerforming analysis {args.type}")

  if args.type == "compression-matrix":
    compression_matrix.run(args)