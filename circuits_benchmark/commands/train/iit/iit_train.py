import argparse
import json
import os
from argparse import Namespace
import pickle

import torch as t
import wandb

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.iit import train_model


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit")
    add_common_args(parser)

    parser.add_argument(
        "--residual-stream-compression-size",
        type=str,
        default="auto",
        help="A list of comma separated sizes for the compressed residual stream, or 'auto' to find the "
        "optimal size.",
    )
    parser.add_argument(
        "--auto-compression-accuracy",
        type=float,
        default=0.95,
        help="The desired test accuracy when using 'auto' compression size.",
    )

    parser.add_argument(
        "-iit", "--iit_weight", type=float, default=1.0, help="IIT weight"
    )
    parser.add_argument(
        "-b",
        "--behavior_weight",
        type=float,
        default=1.0,
        help="Behavior weight",
    )
    parser.add_argument(
        "-s", "--strict_weight", type=float, default=0.4, help="Strict weight"
    )
    parser.add_argument(
        "--wandb_entity", type=str, required=False, help="Wandb entity"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run a wandb sweep"
    )
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb")
    parser.add_argument(
        "--wandb-suffix", type=str, default="", help="Wandb suffix"
    )


def config_is_bad(config):
    iit_weight = config.iit_weight
    behavior_weight = config.behavior_weight
    strict_weight = config.strict_weight

    if iit_weight == behavior_weight and behavior_weight == strict_weight:
        return True
    # reject any combination of [0, x, x]
    weights_tuple = (iit_weight, behavior_weight, strict_weight)
    if weights_tuple in [(0, x, x) for x in [0, 0.5, 1.5]] + [
        (x, 0, x) for x in [0, 0.5, 1.5]
    ] + [(x, x, 0) for x in [0, 0.5, 1.5]]:
        return True

    # reject any combination of [x, 0, 0]
    if weights_tuple in [(x, 0, 0) for x in [0, 0.5, 1.5]] + [
        (0, x, 0) for x in [0, 0.5, 1.5]
    ] + [(0, 0, x) for x in [0, 0.5, 1.5]]:
        return True

    return False


def run_iit_train(case: BenchmarkCase, args: Namespace):
    # if not case.supports_causal_masking():
    #     raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

    tracr_output = case.get_tracr_output()
    hl_model = case.build_transformer_lens_model(
        remove_extra_tensor_cloning=False
    )

    use_wandb = args.use_wandb
    output_dir = args.output_dir

    def main():
        wandb.init()
        if config_is_bad(wandb.config):
            return
        config = {
            **wandb.config,
            "wandb_suffix": args.wandb_suffix,
        }
        train_model(config, case, tracr_output, hl_model, use_wandb=True)

    if args.sweep:
        sweep_config = {
            "name": "tracr_iit",
            "method": "grid",
            "parameters": {
                "atol": {"values": [0.05]},
                "lr": {"values": [1e-3, 1e-4, 1e-5]},
                "use_single_loss": {"values": [True, False]},
                "iit_weight": {"values": [0.5, 1.0, 1.5]},
                "behavior_weight": {"values": [0.5, 1.0, 1.5]},
                "strict_weight": {"values": [0.0, 0.5, 1.0, 1.5]},
                "epochs": {"values": [50]},
                "act_fn": {"values": ["relu", "gelu"]},
            },
        }
        sweep_id = wandb.sweep(
            sweep_config,
            project="iit",
            entity=args.wandb_entity,
        )
        wandb.agent(sweep_id, main)
    else:
        config = {
            "atol": 0.05,
            "lr": 1e-2,
            "use_single_loss": False,
            "iit_weight": args.iit_weight,
            "behavior_weight": args.behavior_weight,
            "strict_weight": args.strict_weight,
            "epochs": 50,
            "act_fn": "gelu",
            "wandb_suffix": args.wandb_suffix,
        }

        args = argparse.Namespace(**config)
        model_pair = train_model(
            args, case, tracr_output, hl_model, use_wandb=use_wandb
        )

        # save the model
        save_dir = f"{output_dir}/ll_models/{case.get_index()}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        weight_int = int(
            args.iit_weight * 10
            + args.behavior_weight * 100
            + args.strict_weight * 1000
        )
        t.save(
            model_pair.ll_model.state_dict(),
            f"{save_dir}/ll_model_{weight_int}.pth",
        )

        # save training args, config
        with open(f"{save_dir}/meta_{weight_int}.json", "w") as f:
            json.dump(config, f)

        # TODO: save the config
        ll_model_cfg = model_pair.ll_model.cfg
        ll_model_cfg_dict = ll_model_cfg.to_dict()
        
        pickle.dump(ll_model_cfg_dict, open(f"{save_dir}/ll_model_cfg_{weight_int}.pkl", "wb"))
