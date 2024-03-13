from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.build_main_parser import build_main_parser
import iit_utils.correspondence as correspondence
from argparse import Namespace
from argparse import ArgumentParser
import circuits_benchmark.commands.algorithms.acdc as acdc
import os
import shutil
from iit_utils.acdc_eval import evaluate_acdc_circuit

parser = ArgumentParser()
parser.add_argument("-t", "--task", type=int, default=3, help="Task number")
case_num = parser.parse_args().task

args = Namespace(
    command="compile",
    indices=f"{case_num}",
    output_dir="/Users/cybershiptrooper/src/interpretability/MATS/circuits-benchmark/results/compile",
    device="cpu",
    seed=1234,
    run_tests=False,
    tests_atol=1e-05,
    fail_on_error=False,
    original_args=["compile", f"-i={case_num}", "-f"],
)
threshold = 0.025
cases = get_cases(args)
case = cases[0]
if not case.supports_causal_masking():
    raise NotImplementedError(f"Case {case.get_index()} does not support causal masking")

tracr_output = case.build_tracr_model()
hl_model = case.build_transformer_lens_model()

metric = "l2" if not hl_model.is_categorical() else "kl"

# this is the graph node -> hl node correspondence
# tracr_hl_corr = correspondence.TracrCorrespondence.from_output(tracr_output)
clean_dirname = f"results/acdc_{case.get_index()}/"
# remove everything in the directory
if os.path.exists(clean_dirname):
    shutil.rmtree(clean_dirname)
cfg_dict = {
    "n_layers": 2,
    "n_heads": 4,
    "d_head": 4,
    "d_model": 8,
    "d_mlp": 16,
    "seed": 0,
    "act_fn": "gelu",
}
ll_cfg = hl_model.cfg.to_dict().copy()
ll_cfg.update(cfg_dict)

ll_model = HookedTracrTransformer(
    ll_cfg, hl_model.tracr_input_encoder, hl_model.tracr_output_encoder, hl_model.residual_stream_labels
)
ll_model.load_weights_from_file(f"ll_models/{case_num}/ll_model_110.pth")

print(ll_model.device)
ll_model.to(ll_model.device)
for param in ll_model.parameters():
    print(param.device)
    break

acdc_args, _ = build_main_parser().parse_known_args(
    [
        "run",
        "acdc",
        f"--threshold={threshold}",
        f"--metric={metric}",
    ]
)  #'--data_size=1000'])

acdc_circuit = acdc.run_acdc(case, acdc_args, ll_model, calculate_fpr_tpr=False)
print("Done running acdc: ")
print(list(acdc_circuit.nodes), list(acdc_circuit.edges))

# get the ll -> hl correspondence
hl_ll_corr = correspondence.TracrCorrespondence.from_output(case=case, tracr_output=tracr_output)
print("hl_ll_corr:", hl_ll_corr)
hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")
# evaluate the acdc circuit
print("Calculating FPR and TPR for threshold", threshold)
result = evaluate_acdc_circuit(acdc_circuit, ll_model, hl_ll_corr, verbose=False)

# save the result
with open(f"{clean_dirname}/result.txt", "w") as f:
    f.write(str(result))
print(f"Saved result to {clean_dirname}/result.txt")

# metric_name = "l2"
# validation_metric = case.get_validation_metric(metric_name, ll_model, data_size=1200)
# from iit_utils.dataset import TracrIITDataset, create_dataset, get_encoded_input_from_torch_input
# data, _ = create_dataset(case, hl_model, 1200, 0)
# inputs, outputs, _ = get_encoded_input_from_torch_input(zip(*data.base_data[:]), hl_model, ll_model.device)
# # print(f"Validation metric: {validation_metric}", "\n\noutputs:", outputs)
# print(f"Validation metric: {validation_metric(outputs.unsqueeze(-1))}")

# raise
