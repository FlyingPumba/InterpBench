import os
import pickle
import shutil
from argparse import Namespace

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.commands.algorithms.eap import EAPRunner
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.circuit_eval import evaluate_hypothesis_circuit
from circuits_benchmark.utils.iit import make_ll_cfg_for_case
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("iit_eap")
  EAPRunner.add_args_to_parser(parser)

  parser.add_argument(
    "-w",
    "--weights",
    type=str,
    default="510",
    help="IIT, behavior, strict weights",
  )
  parser.add_argument(
    "-wandb", "--using_wandb", action="store_true", help="Use wandb"
  )
  parser.add_argument(
    "--load-from-wandb", action="store_true", help="Load model from wandb"
  )


def run_eap_eval(case: BenchmarkCase, args: Namespace):
  eap_runner = EAPRunner(case, args)

  weights = args.weights

  clean_dirname = prepare_output_dir(case, eap_runner, weights, args)

  print(f"Running EAP evaluation for IIT model on case {case.get_name()}")
  print(f"Output directory: {clean_dirname}")

  hl_ll_corr, ll_model = get_ll_model(case, weights, args)

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
  print(
    f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl"
  )
  if args.using_wandb:
    import wandb
    wandb.init(project=f"circuit_discovery",
               group=f"eap_{case.get_name()}_{args.weights}",
               name=f"{args.threshold}")
    wandb.save(f"{clean_dirname}/*", base_path=args.output_dir)

  return result


def get_ll_model(case: TracrBenchmarkCase,
                 weights: str,
                 args: Namespace):
  tracr_output = case.get_tracr_output()

  hl_model = case.get_hl_model()

  ll_cfg = make_ll_cfg_for_case(hl_model, case.get_name())
  ll_model = HookedTracrTransformer(
    ll_cfg,
    hl_model.tracr_input_encoder,
    hl_model.tracr_output_encoder,
    hl_model.residual_stream_labels,
  )

  hl_ll_corr = TracrCorrespondence.from_output(
    case=case, tracr_output=tracr_output
  )

  if weights != "tracr":
    if args.load_from_wandb:
      load_model_from_wandb(case.get_name(), weights, args.output_dir)
    ll_model.load_weights_from_file(
      f"{args.output_dir}/ll_models/{case.get_name()}/ll_model_{weights}.pth"
    )

  ll_model.eval()
  return hl_ll_corr, ll_model


def prepare_output_dir(case, runner, weights, args):
  if runner.edge_count is not None:
    output_suffix = f"weight_{weights}/edge_count_{runner.edge_count}"
  else:
    output_suffix = f"weight_{weights}/threshold_{runner.threshold}"

  clean_dirname = f"{args.output_dir}/eap_{case.get_name()}/{output_suffix}"

  # remove everything in the directory
  if os.path.exists(clean_dirname):
    shutil.rmtree(clean_dirname)

  # mkdir
  os.makedirs(clean_dirname, exist_ok=True)

  return clean_dirname
