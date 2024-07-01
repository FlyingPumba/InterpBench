import os
import pickle
import shutil
from argparse import Namespace

import circuits_benchmark.commands.algorithms.acdc as acdc
import circuits_benchmark.utils.iit.correspondence as correspondence
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.common_args import add_common_args
from circuits_benchmark.utils.circuit_eval import evaluate_hypothesis_circuit
from circuits_benchmark.utils.iit.ll_model_loader import ModelType, get_ll_model


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit_acdc")
    add_common_args(parser)

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
        "--tracr", action="store_true", help="Use tracr model instead of SIIT model"
    )
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Use naturally trained model, instead of SIIT model. This assumes that the model is already trained and stored in <output_dir>/ll_models/<case_index>/ll_model_natural.pth (run train iit for this)",
    )
    parser.add_argument(
        "--load-from-wandb", action="store_true", help="Load model from wandb"
    )
    parser.add_argument(
        "--interp-bench", action="store_true", help="Use interp bench model"
    )
    parser.add_argument(
        "--same-size", action="store_true", help="Use same size for ll model"
    )


def run_acdc_eval(case: BenchmarkCase, args: Namespace):
    threshold = args.threshold
    using_wandb = args.using_wandb

    tracr_output = case.get_tracr_output()
    hl_model = case.build_transformer_lens_model(
        tracr_model=tracr_output.model,
        remove_extra_tensor_cloning=False
    )

    metric = "l2" if not hl_model.is_categorical() else "kl"

    # this is the graph node -> hl node correspondence
    # tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
    model_type = ModelType.make_model_type(args.natural, args.tracr, args.interp_bench)
    weights = ModelType.get_weight_for_model_type(model_type, task=case.get_index())

    output_suffix = f"{weights}/threshold_{threshold}"
    clean_dirname = f"{args.output_dir}/acdc_{case.get_index()}/{output_suffix}"
    # remove everything in the directory
    if os.path.exists(clean_dirname):
        shutil.rmtree(clean_dirname)

    wandb_str = "--using-wandb" if using_wandb else ""
    from circuits_benchmark.commands.build_main_parser import build_main_parser

    acdc_args, _ = build_main_parser().parse_known_args(
        [
            "run",
            "acdc",
            f"--threshold={threshold}",
            f"--metric={metric}",
            wandb_str,
            "--wandb-entity-name=cybershiptrooper",
            f"--wandb-project-name=acdc_{case.get_index()}_{model_type}",
        ]
    )  #'--data_size=1000'])

    if model_type == ModelType.TRACR:
        acdc_circuit, result = acdc.run_acdc(
            case, acdc_args, calculate_fpr_tpr=True, output_suffix=output_suffix
        )
    else:
        # load the ll model
        hl_ll_corr, ll_model = get_ll_model(case=case, model_type=model_type, load_from_wandb=args.load_from_wandb, device=args.device, output_dir=args.output_dir, same_size=args.same_size)

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

        print("hl_ll_corr:", hl_ll_corr)
        hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")
        # evaluate the acdc circuit
        print("Calculating FPR and TPR for threshold", threshold)
        result = evaluate_hypothesis_circuit(
            acdc_circuit,
            ll_model,
            hl_ll_corr,
            case,
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
            group=f"acdc_{case.get_index()}_{weights}",
            name=f"{args.threshold}",
        )
        wandb.save(f"{clean_dirname}/*", base_path=args.output_dir)
    return result
