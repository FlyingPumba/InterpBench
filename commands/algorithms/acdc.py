import datetime
import gc
import os
import shutil

import torch
import wandb

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_graphics import show
from benchmark.benchmark_case import BenchmarkCase
from commands.compilation.compile_benchmark import build_transformer_lens_model
from utils.project_paths import get_default_output_dir


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("acdc")
  parser.add_argument("-i", "--indices", type=str, default=None,
                      help="A list of comma separated indices of the cases to run against. "
                           "If not specified, all cases will be run.")
  parser.add_argument("-f", "--force", action="store_true",
                      help="Force compilation of cases, even if they have already been compiled.")
  parser.add_argument("-o", "--output-dir", type=str, default=get_default_output_dir(),
                      help="The directory to save the results to.")

  parser.add_argument('--threshold', type=float, required=True, help='Value for threshold')
  parser.add_argument('--metric', type=str, required=True, choices=["kl", "l2"],
                      help="Which metric to use for the experiment")

  parser.add_argument('--first-cache-cpu', type=str, required=False, default="True",
                      help='Value for first_cache_cpu (the old name for the `online_cache`)')
  parser.add_argument('--second-cache-cpu', type=str, required=False, default="True",
                      help='Value for second_cache_cpu (the old name for the `corrupted_cache`)')
  parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
  parser.add_argument('--using-wandb', action='store_true', help='Use wandb')
  parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock",
                      help='Value for wandb_entity_name')
  parser.add_argument('--wandb-group-name', type=str, required=False, default="default",
                      help='Value for wandb_group_name')
  parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc",
                      help='Value for wandb_project_name')
  parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for wandb_run_name')
  parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
  parser.add_argument("--wandb-mode", type=str, default="online")
  parser.add_argument('--indices-mode', type=str, default="normal")
  parser.add_argument('--names-mode', type=str, default="normal")
  parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="The device to use for ACDC.")
  parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument("--max-num-epochs", type=int, default=100_000)
  parser.add_argument('--single-step', action='store_true', help='Use single step, mostly for testing')
  parser.add_argument("--abs-value-threshold", action='store_true',
                      help='Use the absolute value of the result to check threshold')


def run_acdc(case: BenchmarkCase, args):
  tl_model = build_transformer_lens_model(case, force=args.force, device=args.device)

  output_dir = args.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  images_output_dir = os.path.join(output_dir, "images")
  if not os.path.exists(images_output_dir):
    os.makedirs(images_output_dir)

  # Check that dot program is in path
  if not shutil.which("dot"):
    raise ValueError("dot program not in path, cannot generate graphs for ACDC.")

  if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
  torch.manual_seed(args.seed)

  if args.first_cache_cpu is None:
    online_cache_cpu = True
  elif args.first_cache_cpu.lower() == "false":
    online_cache_cpu = False
  elif args.first_cache_cpu.lower() == "true":
    online_cache_cpu = True
  else:
    raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")

  if args.second_cache_cpu is None:
    corrupted_cache_cpu = True
  elif args.second_cache_cpu.lower() == "false":
    corrupted_cache_cpu = False
  elif args.second_cache_cpu.lower() == "true":
    corrupted_cache_cpu = True
  else:
    raise ValueError(f"second_cache_cpu must be either True or False, got {args.second_cache_cpu}")

  threshold = args.threshold  # only used if >= 0.0
  metric_name = args.metric
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

  second_metric = None  # some tasks only have one metric
  use_pos_embed = True  # Always true for all tracr models.

  validation_metric = case.get_validation_metric(metric_name, tl_model)
  toks_int_values = case.get_clean_data().get_inputs()
  toks_int_values_other = case.get_corrupted_data().get_inputs()

  try:
    with open(__file__, "r") as f:
      notes = f.read()
  except Exception as e:
    notes = "No notes generated, expected when running in an .ipynb file. Error is " + str(e)

  tl_model.reset_hooks()

  # Save some mem
  gc.collect()
  torch.cuda.empty_cache()

  # Setup wandb if needed
  if wandb_run_name is None:
    wandb_run_name = f"{'_randomindices' if indices_mode == 'random' else ''}_{threshold}{'_zero' if zero_ablation else ''}"
  else:
    assert wandb_run_name is not None, "I want named runs, always"

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
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=zero_ablation,
    abs_value_threshold=args.abs_value_threshold,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
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
    print(exp.count_no_edges())

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

  if using_wandb:
    edges_fname = f"edges.pth"
    exp.save_edges(edges_fname)
    artifact = wandb.Artifact(edges_fname, type="dataset")
    artifact.add_file(edges_fname)
    wandb.log_artifact(artifact)
    os.remove(edges_fname)
    wandb.finish()

  exp.save_subgraph(
    fpath=f"{output_dir}/subgraph.pth",
    return_it=True,
  )
