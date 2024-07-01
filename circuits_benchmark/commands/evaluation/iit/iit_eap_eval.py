import os
import pickle
import shutil
from argparse import Namespace

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.commands.algorithms.eap import EAPRunner
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader_from_args


def setup_args_parser(subparsers):
    parser = subparsers.add_parser("iit_eap")
    EAPRunner.add_args_to_parser(parser)

    parser.add_argument(
        "--same-size", action="store_true", help="Use same size for ll model"
    )


def run_eap_eval(case: BenchmarkCase, args: Namespace):
    eap_runner = EAPRunner(case, args)

    ll_model_loader = get_ll_model_loader_from_args(case, args)
    clean_dirname = prepare_output_dir(case, eap_runner, ll_model_loader, args)

    print(f"Running EAP evaluation for IIT model on case {case.get_name()}")
    print(f"Output directory: {clean_dirname}")

    hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
        load_from_wandb=args.load_from_wandb,
        device=args.device,
        output_dir=args.output_dir,
        same_size=args.same_size,
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


def prepare_output_dir(case, runner, ll_model_loader, args):
  if runner.edge_count is not None:
    output_suffix = f"{ll_model_loader.get_output_suffix()}/edge_count_{runner.edge_count}"
  else:
    output_suffix = f"{ll_model_loader.get_output_suffix()}/threshold_{runner.threshold}"

  clean_dirname = f"{args.output_dir}/eap_{case.get_name()}/{output_suffix}"

  # remove everything in the directory
  if os.path.exists(clean_dirname):
    shutil.rmtree(clean_dirname)

  # mkdir
  os.makedirs(clean_dirname, exist_ok=True)

  return clean_dirname
