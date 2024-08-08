# %%
from argparse import ArgumentParser

import torch

import iit.model_pairs as mp
import interp_utils.lens.logit_lens as logit_lens
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
    get_ll_model_loader,
)
from interp_utils.lens.plot_utils import (
    get_formatted_node_names_in_circuit,
    plot_combined_pearson,
    plot_pearson,
    plot_pearson_at_vocab_idx,
    save_lens_results,
    plot_explained_variance_combined
)

parser = ArgumentParser()
parser.add_argument("--task", type=str, default="3")
parser.add_argument("--max_len", type=int, default=1000)
task_idx = parser.parse_args().task
max_len = parser.parse_args().max_len

task: BenchmarkCase = get_cases(indices=[task_idx])[0]

ll_model_loader = get_ll_model_loader(task, interp_bench=True)
hl_ll_corr, model = ll_model_loader.load_ll_model_and_correspondence(
    device="cuda" if torch.cuda.is_available() else "cpu"
)
# turn off grads
model.eval()
model.requires_grad_(False)

hl_model = task.get_hl_model()
model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)

# %%
unique_test_data = task.get_clean_data(max_samples=max_len, unique_data=True)

loader = torch.utils.data.DataLoader(
    unique_test_data, batch_size=256, shuffle=False, drop_last=False
)

# %%
if model_pair.hl_model.is_categorical():
    # preprocess model for logit lens
    model.center_writing_weights(state_dict=model.state_dict())
    model.center_unembed(state_dict=model.state_dict())
    model.refactor_factored_attn_matrices(state_dict=model.state_dict())
try:
    model.fold_layer_norm(state_dict=model.state_dict())
except:  # noqa: E722
    print("No layer norm to fold")

# %%

logit_lens_results, labels = logit_lens.do_logit_lens(model_pair, loader)
save_lens_results(
    logit_lens_results, 
    labels, 
    nodes_in_circuit=get_formatted_node_names_in_circuit(model_pair),
    case=task,
    tuned_lens=False,
)

# %%
nodes = get_formatted_node_names_in_circuit(model_pair)

# %%
for k in logit_lens_results.keys():
    plot_pearson(
        key=k,
        lens_results=logit_lens_results,
        labels=labels,
        is_categorical=model_pair.hl_model.is_categorical(),
        in_circuit=k in nodes,
        tuned_lens=False,
        case_name=task.get_name(),
        show=False,
    )

# %%
plot_combined_pearson(
    lens_results=logit_lens_results,
    labels=labels,
    nodes_in_circuit=get_formatted_node_names_in_circuit(model_pair),
    is_categorical=model_pair.hl_model.is_categorical(),
    tuned_lens=False,
    case_name=task.get_name(),
)


# %%
if model_pair.hl_model.is_categorical():
    logit_lens_per_vocab, per_vocab_labels = logit_lens.do_logit_lens_per_vocab_idx(
        model_pair, loader
    )
    for k in logit_lens_per_vocab.keys():
        for i in logit_lens_per_vocab[k].keys():
            plot_pearson_at_vocab_idx(
                key=k,
                vocab_idx=i,
                lens_results_per_vocab=logit_lens_per_vocab,
                per_vocab_labels=per_vocab_labels,
                in_circuit=k in nodes,
                tuned_lens=False,
                case_name=task.get_name(),
                show=False,
            )

# %%

plot_explained_variance_combined(
    lens_results=logit_lens_results,
    labels=labels,
    nodes_in_circuit=nodes,
    is_categorical=model_pair.hl_model.is_categorical(),
    tuned_lens=False,
    case_name=task.get_name(),
)
