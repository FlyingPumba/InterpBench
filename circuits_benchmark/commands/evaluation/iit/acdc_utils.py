import datetime
import gc
import os
import random
import shutil
import sys

import numpy as np
import torch
import wandb
from torch.nn import init

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_graphics import show
from circuits_benchmark.commands.analysis.acdc_circuit import calculate_fpr_and_tpr
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from argparse import Namespace
from copy import deepcopy

class ACDCRunner:
    def __init__(self, task: str, args: Namespace):
        self.task = task
        self.setup_acdc_args(args)

    def setup_acdc_args(self, _args):
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

        acdc_circuit = build_acdc_circuit(exp.corr)
        acdc_circuit.save(f"{output_dir}/final_circuit.pkl")
        return acdc_circuit, exp