from argparse import ArgumentParser
from circuits_benchmark.utils.iit import run_acdc_eval

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--case", type=int, default=3, help="Task number")
    parser.add_argument("-w", "--weights", type=str, default="510", help="IIT, behavior, strict weights")
    parser.add_argument("-t", "--threshold", type=float, default=0.025, help="Threshold for ACDC")
    parser.add_argument("-wandb", "--using_wandb", action="store_true", help="Use wandb")
    args = parser.parse_args()
    print(args)
    case_num = args.case
    weight = args.weights
    threshold = args.threshold
    run_acdc_eval(case_num, weight, threshold, using_wandb=args.using_wandb)
