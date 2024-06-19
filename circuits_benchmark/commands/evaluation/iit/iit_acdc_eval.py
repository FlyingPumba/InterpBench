import os
import pickle
import shutil
from argparse import Namespace

import circuits_benchmark.commands.algorithms.acdc as acdc
import circuits_benchmark.utils.iit.correspondence as correspondence
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.transformers.acdc_circuit_builder import (
    build_acdc_circuit,
)
from circuits_benchmark.transformers.hooked_tracr_transformer import (
    HookedTracrTransformer,
)
from circuits_benchmark.utils.circuit_eval import evaluate_hypothesis_circuit
from circuits_benchmark.utils.iit.ll_cfg import make_ll_cfg_for_case
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit_acdc")
    add_common_args(parser)

    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="510",
        help="IIT, behavior, strict weights",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.025,
        help="Threshold for ACDC",
    )
    parser.add_argument(
        "-wandb", "--using_wandb", action="store_true", help="Use wandb"
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )
    parser.add_argument(
        "--same-size", action="store_true", help="Use same size for ll model"
    )


def run_acdc_eval(case: BenchmarkCase, args: Namespace):
    case_num = case.get_index()

    weight = args.weights
    threshold = args.threshold
    using_wandb = args.using_wandb

    tracr_output = case.get_tracr_output()
    hl_model = case.build_transformer_lens_model(remove_extra_tensor_cloning=False)

    metric = "l2" if not hl_model.is_categorical() else "kl"

    # this is the graph node -> hl node correspondence
    # tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
    output_suffix = f"weight_{weight}/threshold_{threshold}"
    clean_dirname = f"{args.output_dir}/acdc_{case.get_index()}/{output_suffix}"
    # remove everything in the directory
    if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)

    wandb_str = f"--using-wandb" if using_wandb else ""
    from circuits_benchmark.commands.build_main_parser import build_main_parser

    acdc_args, _ = build_main_parser().parse_known_args(
        [
            "run",
            "acdc",
            f"--threshold={threshold}",
            f"--metric={metric}",
            wandb_str,
            "--wandb-entity-name=cybershiptrooper",
            f"--wandb-project-name=acdc_{case.get_index()}_{weight}",
        ]
    )  #'--data_size=1000'])

    if weight == "tracr":
        acdc_circuit, result = acdc.run_acdc(
            case, acdc_args, calculate_fpr_tpr=True, output_suffix=output_suffix
        )
    else:
        # get best weight if needed
        if weight == "best":
            from circuits_benchmark.utils.iit.best_weights import get_best_weight
            weight = get_best_weight(case.get_index())
        
        # load from wandb if needed
        if args.load_from_wandb:
            load_model_from_wandb(
                case_num, weight, args.output_dir, same_size=args.same_size
            )
        # make ll cfg
        try:
            ll_cfg = pickle.load(
                open(
                    f"{args.output_dir}/ll_models/{case.get_index()}/ll_model_cfg_{weight}.pkl",
                    "rb",
                )
            )
        except FileNotFoundError:
            ll_cfg = make_ll_cfg_for_case(
                hl_model, case.get_index(), same_size=args.same_size
            )

        # make ll model
        ll_model = HookedTracrTransformer(
            ll_cfg,
            hl_model.tracr_input_encoder,
            hl_model.tracr_output_encoder,
            hl_model.residual_stream_labels,
            remove_extra_tensor_cloning=False,
        )
        ll_model.load_weights_from_file(
            f"{args.output_dir}/ll_models/{case_num}/ll_model_{weight}.pth"
        )

        ll_model.to(args.device)

        # run acdc
        acdc_circuit, acdc_result = acdc.run_acdc(
            case,
            acdc_args,
            ll_model,
            calculate_fpr_tpr=False,
            output_suffix=output_suffix,
        )
        print("Done running acdc: ")
        print(list(acdc_circuit.nodes), list(acdc_circuit.edges))

        # get the ll -> hl correspondence
        if args.same_size:
            hl_ll_corr = correspondence.TracrCorrespondence.make_identity_corr(
                tracr_output=tracr_output
            )
        else:
            hl_ll_corr = correspondence.TracrCorrespondence.from_output(
                case, tracr_output
            )
        print("hl_ll_corr:", hl_ll_corr)
        hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")
        # evaluate the acdc circuit
        print("Calculating FPR and TPR for threshold", threshold)
        from acdc.TLACDCCorrespondence import TLACDCCorrespondence

        full_corr = TLACDCCorrespondence.setup_from_model(ll_model, use_pos_embed=True)
        full_circuit = build_acdc_circuit(full_corr)
        result = evaluate_hypothesis_circuit(
            acdc_circuit,
            ll_model,
            hl_ll_corr,
            case,
            full_circuit=full_circuit,
            verbose=False,
        )
        result.update(acdc_result)

    # save the result
    with open(f"{clean_dirname}/result.txt", "w") as f:
        f.write(str(result))
    pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
    print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")
    if args.using_wandb:
        import wandb

        wandb.init(
            project=f"circuit_discovery{'_same_size' if args.same_size else ''}",
            group=f"acdc_{case.get_index()}_{args.weights}",
            name=f"{args.threshold}",
        )
        wandb.save(f"{clean_dirname}/*", base_path=args.output_dir)
    return result
