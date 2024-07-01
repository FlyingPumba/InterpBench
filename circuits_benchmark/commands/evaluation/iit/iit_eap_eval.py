import os
import pickle
import shutil
from argparse import Namespace

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.algorithms.eap import EAPRunner
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit
from circuits_benchmark.utils.iit.ll_model_loader import ModelType, get_ll_model


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit_eap")
    EAPRunner.add_args_to_parser(parser)

    parser.add_argument(
        "--tracr", action="store_true", help="Use tracr model instead of SIIT model"
    )
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Use naturally trained model, instead of SIIT model. This assumes that the model is already trained and stored in wandb or <output_dir>/ll_models/<case_index>/ll_model_natural.pth (run train iit for this)",
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


def run_eap_eval(case: BenchmarkCase, args: Namespace):
    eap_runner = EAPRunner(case, args)
    model_type = ModelType.make_model_type(args.natural, args.tracr, args.interp_bench)
    clean_dirname = prepare_output_dir(case, eap_runner, model_type, args)

    print(f"Running EAP evaluation for IIT model on case {case.get_name()}")
    print(f"Output directory: {clean_dirname}")
    
    hl_ll_corr, ll_model = get_ll_model(
        case,
        model_type,
        args.load_from_wandb,
        args.device,
        args.output_dir,
        args.same_size,
    )

    clean_dataset = case.get_clean_data(max_samples=args.data_size)
    corrupted_dataset = case.get_corrupted_data(max_samples=args.data_size)
    eap_circuit = eap_runner.run(ll_model, clean_dataset, corrupted_dataset)

    print("hl_ll_corr:", hl_ll_corr)
    hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")

    print("Calculating FPR and TPR")
    result = evaluate_hypothesis_circuit(
        eap_circuit,
        ll_model,
        hl_ll_corr,
        case,
        verbose=False,
        use_embeddings=False,
    )

    # save the result
    with open(f"{clean_dirname}/result.txt", "w") as f:
        f.write(str(result))

    pickle.dump(result, open(f"{clean_dirname}/result.pkl", "wb"))
    print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")
    if args.using_wandb:
        import wandb

        wandb.init(
            project="circuit_discovery",
            group=f"eap_{case.get_name()}_{args.weights}",
            name=f"{args.threshold}",
        )
        wandb.save(f"{clean_dirname}/*", base_path=args.output_dir)

    return result


def prepare_output_dir(case, runner, model_type, args):
  weight = ModelType.get_weight_for_model_type(model_type, task=case.get_name())
  if runner.edge_count is not None:
    output_suffix = f"weight_{weight}/edge_count_{runner.edge_count}"
  else:
    output_suffix = f"weight_{weight}/threshold_{runner.threshold}"

  clean_dirname = f"{args.output_dir}/eap_{case.get_name()}/{output_suffix}"

  # remove everything in the directory
  if os.path.exists(clean_dirname):
    shutil.rmtree(clean_dirname)

  # mkdir
  os.makedirs(clean_dirname, exist_ok=True)

  return clean_dirname
