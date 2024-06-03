from iit.utils.eval_scripts import eval_ioi
import torch

def setup_args_parser(subparsers):
    parser = subparsers.add_parser("ioi")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="100_100_40",
        help="IIT_behavior_strict weights",
    )
    parser.add_argument("-m", "--mean", type=bool, default=True, help="Use mean cache")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for making mean cache (if using mean ablation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-samples", type=int, default=18000, help="Number of samples"
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )
    parser.add_argument(
        "--next-token", action="store_true", help="Use next token model"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
    )

def run_eval_ioi(args):
    eval_ioi(args)
