import argparse
import datetime
import os
from argparse import Namespace
from copy import deepcopy

import torch

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_graphics import show
from circuits_benchmark.utils.circuit.circuit_eval import build_from_acdc_correspondence


class ACDCRunner:
    def __init__(self, task: str, args: Namespace):
        self.task = task
        self.configure_acdc(args)

    @staticmethod
    def add_args_to_parser(parser):
        parser.add_argument(
            "-w",
            "--weights",
            type=str,
            default="100_100_40",
            help="IIT, behavior, strict weights",
        )
        parser.add_argument(
            "--output-dir", type=str, default="./results", help="Output directory"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use", 
        )
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.025,
            help="Threshold for ACDC",
        )
        parser.add_argument("--data-size", type=int, required=False, default=1000, help="How many samples to use")
        parser.add_argument(
            "-wandb", "--using_wandb", action="store_true", help="Use wandb"
        )
        parser.add_argument(
            "--load-from-wandb", action="store_true", help="Load model from wandb"
        )
        parser.add_argument(
            "--include-mlp", action="store_true", help="Evaluate group 'with_mlp'"
        )
        parser.add_argument(
            "--next-token", action="store_true", help="Use next token model"
        )
        parser.add_argument(
            "--use-pos-embed", action="store_true", help="Use positional embeddings"
        )

        parser.add_argument(
            "--first-cache-cpu",
            type=str,
            required=False,
            default="True",
            help="Value for first_cache_cpu (the old name for the `online_cache`)",
        )
        parser.add_argument(
            "--second-cache-cpu",
            type=str,
            required=False,
            default="True",
            help="Value for second_cache_cpu (the old name for the `corrupted_cache`)",
        )
        parser.add_argument("--zero-ablation", action="store_true", help="Use zero ablation")
        parser.add_argument("--using-wandb", action="store_true", help="Use wandb")
        parser.add_argument(
            "--wandb-entity-name",
            type=str,
            required=False,
            default="remix_school-of-rock",
            help="Value for wandb_entity_name",
        )
        parser.add_argument(
            "--wandb-group-name", type=str, required=False, default="default", help="Value for wandb_group_name"
        )
        parser.add_argument(
            "--wandb-project-name", type=str, required=False, default="acdc", help="Value for wandb_project_name"
        )
        parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Value for wandb_run_name")
        parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
        parser.add_argument("--wandb-mode", type=str, default="online")
        parser.add_argument("--indices-mode", type=str, default="normal")
        parser.add_argument("--names-mode", type=str, default="normal")
        parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")
        parser.add_argument("--max-num-epochs", type=int, default=100_000)
        parser.add_argument("--single-step", action="store_true", help="Use single step, mostly for testing")
        parser.add_argument(
            "--abs-value-threshold", action="store_true", help="Use the absolute value of the result to check threshold"
        )

    def configure_acdc(self, _args):
        args = deepcopy(_args)
        if args.first_cache_cpu is None:
            self.online_cache_cpu = True
        elif args.first_cache_cpu.lower() == "false":
            self.online_cache_cpu = False
        elif args.first_cache_cpu.lower() == "true":
            self.online_cache_cpu = True
        else:
            raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")

        if args.second_cache_cpu is None:
            self.corrupted_cache_cpu = True
        elif args.second_cache_cpu.lower() == "false":
            self.corrupted_cache_cpu = False
        elif args.second_cache_cpu.lower() == "true":
            self.corrupted_cache_cpu = True
        else:
            raise ValueError(f"second_cache_cpu must be either True or False, got {args.second_cache_cpu}")
        
        args.output_dir = os.path.join(args.output_dir, f"acdc_{self.task}", f"weight_{args.weights}", f"threshold_{args.threshold}")
        args.images_output_dir = os.path.join(args.output_dir, "images")
        os.makedirs(args.images_output_dir, exist_ok=True)
        self.args = args


    def run_acdc(self, tl_model, 
                 clean_dataset, 
                 corrupt_dataset, 
                 validation_metric, 
                 second_metric=None,):
        args = self.args
        corrupted_cache_cpu = self.corrupted_cache_cpu
        online_cache_cpu = self.online_cache_cpu

        threshold = args.threshold  # only used if >= 0.0
        zero_ablation = True if args.zero_ablation else False
        using_wandb = True if args.using_wandb else False
        wandb_entity_name = args.wandb_entity_name
        wandb_project_name = args.wandb_project_name
        wandb_run_name = args.wandb_run_name
        wandb_group_name = args.wandb_group_name
        indices_mode = args.indices_mode
        names_mode = args.names_mode
        device = args.device
        single_step = True if args.single_step else False
        output_dir = args.output_dir
        images_output_dir = args.images_output_dir

        use_pos_embed = args.use_pos_embed

        tl_model.reset_hooks()
        exp = TLACDCExperiment(
            model=tl_model,
            threshold=threshold,
            images_output_dir=images_output_dir,
            using_wandb=using_wandb,
            wandb_entity_name=wandb_entity_name,
            wandb_project_name=wandb_project_name,
            wandb_run_name=wandb_run_name,
            wandb_group_name=wandb_group_name,
            wandb_dir=args.wandb_dir,
            wandb_mode=args.wandb_mode,
            zero_ablation=zero_ablation,
            abs_value_threshold=args.abs_value_threshold,
            ds=clean_dataset,
            ref_ds=corrupt_dataset,
            metric=validation_metric,
            second_metric=second_metric,
            verbose=True,
            indices_mode=indices_mode,
            names_mode=names_mode,
            corrupted_cache_cpu=corrupted_cache_cpu,
            hook_verbose=False,
            online_cache_cpu=online_cache_cpu,
            add_sender_hooks=True,
            use_pos_embed=use_pos_embed,
            add_receiver_hooks=False,
            remove_redundant=False,
            show_full_index=use_pos_embed,
        )

        exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in range(args.max_num_epochs):
            exp.step(testing=False)

            show(
                exp.corr,
                fname=f"{images_output_dir}/img_new_{i + 1}.png",
            )

            print(i, "-" * 50)
            print(exp.count_num_edges())

            if i == 0:
                exp.save_edges(os.path.join(output_dir, "edges.pkl"))

            if exp.current_node is None or single_step:
                show(
                    exp.corr,
                    fname=f"{images_output_dir}/ACDC_new_{exp_time}.png",
                    show_placeholders=True,
                )
                break

        exp.save_edges(os.path.join(output_dir, "another_final_edges.pkl"))

        exp.save_subgraph(
            fpath=f"{output_dir}/subgraph.pth",
            return_it=True,
        )

        acdc_circuit = build_from_acdc_correspondence(exp.corr)
        acdc_circuit.save(f"{output_dir}/final_circuit.pkl")
        return acdc_circuit, exp
    
    @classmethod
    def make_default_runner(cls, task: str):
        parser = argparse.ArgumentParser()
        cls.add_args_to_parser(parser)
        args = parser.parse_args([])
        return cls(task, args)
