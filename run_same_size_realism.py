import os
import argparse
from circuits_benchmark.utils.iit.best_weights import get_best_weight

thresholds = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.025,
    0.05,
    0.1,
    0.2,
    0.5,
    0.8,
    1.0,
    10.0,
    20.0,
    50.0,
    100.0,
]

def run_all_same_size_realism_for_case(case, disable_wandb = False):
    if disable_wandb:
        iit_command_template = """python main.py train iit -i {} --same-size -iit {} -s {} -b {} --epochs 500 --use-wandb"""
        acdc_command_template = """python main.py eval iit_acdc -i {} -w {} -t {} --abs-value-threshold --same-size"""
        circuit_score_command_template = """python main.py eval node_realism -i {} --mean --relative 1 -w {} -t {} --same-size"""
    else:
        iit_command_template = """python main.py train iit -i {} --same-size -iit {} -s {} -b {} --epochs 500 --use-wandb --save-model-wandb"""
        acdc_command_template = """python main.py eval iit_acdc -i {} -w {} -t {} --abs-value-threshold --same-size -wandb --load-from-wandb """
        circuit_score_command_template = """python main.py eval node_realism -i {} --mean --relative 1 -w {} -t {} --same-size --use-wandb --load-from-wandb """

    best_weights_for_case = get_best_weight(case, individual=True)
    best_weight_combined = get_best_weight(case, individual=False)
    
    # best
    os.system(iit_command_template.format(case, best_weights_for_case["iit"], best_weights_for_case["strict"], best_weights_for_case["behavior"]) + " --save-model-wandb")
    # natural
    os.system(f"python main.py train iit -i {case} --same-size -iit 0 -s 0 -b 1 --save-model-wandb")

    # run acdc
    for threshold in thresholds:
        os.system(acdc_command_template.format(case, best_weight_combined, threshold))
        os.system(circuit_score_command_template.format(case, best_weight_combined, threshold))

        os.system(acdc_command_template.format(case, "100", threshold))
        os.system(circuit_score_command_template.format(case, "100", threshold))

    os.system("source clean_acdc_files.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--case", type=int, required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run_all_same_size_realism_for_case(args.case, args.no_wandb)