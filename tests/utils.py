import os

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train import train
from circuits_benchmark.utils.project_paths import get_default_output_dir


def setup_iit_models():
    # detect if SIIT and Natural model for Case 3 are available, and train them if not
    output_dir = get_default_output_dir()

    natural_model_path = f"{output_dir}/ll_models/3/ll_model_100.pth"
    if not os.path.exists(natural_model_path):
        # train natural model
        args, _ = build_main_parser().parse_known_args(["train",
                                                        "iit",
                                                        "-i=3",
                                                        "--epochs=0",
                                                        "--early-stop",
                                                        "-s=0",
                                                        "-iit=0",
                                                        "--num-samples=10",
                                                        "--device=cpu"])
        train.run(args)
    assert os.path.exists(natural_model_path)

    siit_model_path = f"{output_dir}/ll_models/3/ll_model_510.pth"
    if not os.path.exists(siit_model_path):
        # train SIIT model
        args, _ = build_main_parser().parse_known_args(["train",
                                                        "iit",
                                                        "-i=3",
                                                        "--epochs=0",
                                                        "--early-stop",
                                                        "--num-samples=10",
                                                        "--device=cpu"])
        train.run(args)
    assert os.path.exists(siit_model_path)