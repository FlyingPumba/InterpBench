import gc
import os
import pickle
import random
from typing import Callable
from typing import Optional

import numpy as np
import torch
import wandb
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeType
from acdc.docstring.utils import AllDataThings
from subnetwork_probing.masked_transformer import EdgeLevelMaskedTransformer
from tqdm import tqdm


def save_edges(corr: TLACDCCorrespondence, fname: str):
    edges_list = []
    for t, e in corr.edge_dict().items():
        if e.present and e.edge_type != EdgeType.PLACEHOLDER:
            edges_list.append((t, e.effect_size))

    with open(fname, "wb") as f:
        pickle.dump(edges_list, f)

    print(f"Saved edges to {fname}")


def train_edge_sp(
    args,
    masked_model: EdgeLevelMaskedTransformer,
    all_task_things: AllDataThings,
    print_every: int = 100,
    eval_fn: Optional[Callable] = None,
):
    print(f"Using memory {torch.cuda.memory_allocated():_} bytes at training start")
    epochs = args.epochs
    lambda_reg = args.lambda_reg

    # Set the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.using_wandb:
        wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=args,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
        )

    test_metric_fns = all_task_things.test_metrics
    print(args)
    gc.collect()
    masked_model.freeze_weights()

    # one parameter per thing that is masked
    mask_params = list(p for p in masked_model.mask_parameter_list if p.requires_grad)
    # parameters for the probe (we don't use a probe)
    # model_params = list(p for p in masked_model.model.parameters() if p.requires_grad)
    # assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=args.lr)

    print(f"Using memory {torch.cuda.memory_allocated():_} bytes after optimizer init")
    if args.zero_ablation:
        validation_patch_data = test_patch_data = None
    else:
        validation_patch_data = all_task_things.validation_patch_data
        test_patch_data = all_task_things.test_patch_data

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        masked_model.train()
        trainer.zero_grad()
        with masked_model.with_fwd_hooks_and_new_ablation_cache(validation_patch_data) as hooked_model:
            # print(f"Using memory {torch.cuda.memory_allocated():_} bytes before forward")
            metric_loss = all_task_things.validation_metric(hooked_model(all_task_things.validation_data))
            # print(f"Using memory {torch.cuda.memory_allocated():_} bytes after forward")
        regularizer_term = masked_model.regularization_loss()
        loss = metric_loss + regularizer_term * lambda_reg
        loss.backward()

        trainer.step()

        if epoch % print_every == 0 and args.print_stats:
            with torch.no_grad():
                with masked_model.with_fwd_hooks_and_new_ablation_cache(test_patch_data) as hooked_model:
                    test_metric_loss = test_metric_fns["loss"](hooked_model(all_task_things.test_data))
                    test_metrc_acc = test_metric_fns["accuracy"](hooked_model(all_task_things.test_data))
            test_loss = test_metric_loss + regularizer_term * lambda_reg
            corr = masked_model.get_edge_level_correspondence_from_masks()
            result = eval_fn(corr)
            if args.using_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "num_edges": masked_model.num_edges(),
                        "regularization_loss": regularizer_term.item(),
                        "validation_metric_loss": metric_loss.item(),
                        "test_metric_loss": test_metric_loss.item(),
                        "test_metric_accuracy": test_metrc_acc,
                        "total_loss": loss.item(),
                        "test_total_loss": test_loss,
                        "nodes_fpr": result.nodes.fpr,
                        "nodes_tpr": result.nodes.tpr,
                        "edges_fpr": result.edges.fpr,
                        "edges_tpr": result.edges.tpr,
                    }
                )

    # Save edges to create data for plots later
    corr = masked_model.get_edge_level_correspondence_from_masks()
    edges_fname = "edges.pth"  # note this is a pickle file
    wandb_dir = os.environ.get("WANDB_DIR")
    if wandb_dir is None:
        save_edges(corr, edges_fname)
    else:
        save_edges(corr, os.path.join(wandb_dir, edges_fname))
    artifact = wandb.Artifact(edges_fname, type="dataset")
    artifact.add_file(edges_fname)
    if args.using_wandb:
        wandb.log_artifact(artifact)
    os.remove(edges_fname)

    # Now calculate final metrics
    with torch.no_grad():
        # The loss has a lot of variance so let's just average over a few runs with the same seed
        rng_state = torch.random.get_rng_state()

        # Final training loss
        metric_loss = 0.0
        masked_model.calculate_and_store_ablation_cache(
            all_task_things.validation_patch_data if not args.zero_ablation else None
        )

        for _ in range(args.n_loss_average_runs):
            with masked_model.with_fwd_hooks_and_new_ablation_cache(validation_patch_data) as hooked_model:
                metric_loss += all_task_things.validation_metric(hooked_model(all_task_things.validation_data)).item()
        print(f"Final train/validation metric: {metric_loss:.4f}")

        masked_model.calculate_and_store_ablation_cache(
            all_task_things.test_patch_data if not args.zero_ablation else None
        )

        test_specific_metrics = {}
        for k, fn in test_metric_fns.items():
            torch.random.set_rng_state(rng_state)
            test_specific_metric_term = 0.0
            # Test loss
            for _ in range(args.n_loss_average_runs):
                with masked_model.with_fwd_hooks_and_new_ablation_cache(test_patch_data) as hooked_model:
                    test_specific_metric_term += fn(hooked_model(all_task_things.test_data))
                test_specific_metrics[f"test_{k}"] = test_specific_metric_term

        print(f"Final test metric: {test_specific_metrics}")

        log_dict = dict(
            # number_of_nodes=number_of_nodes,
            specific_metric=metric_loss,
            # nodes_to_mask=nodes_to_mask,
            **test_specific_metrics,
        )
    return masked_model, log_dict
