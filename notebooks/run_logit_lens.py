# %%
import pickle
import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases

task = get_cases(indices=['3'])[0]
task_idx = task.get_index()

# %%
dir_name = f"../InterpBench/{task_idx}"
cfg_dict = pickle.load(open(f"{dir_name}/ll_model_cfg.pkl", "rb"))
cfg = HookedTransformerConfig.from_dict(cfg_dict)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer(cfg)
weights = torch.load(f"{dir_name}/ll_model.pth", map_location=cfg.device)
model.load_state_dict(weights)

# %%
# turn off grads
model.eval()
model.requires_grad_(False)
torch.set_grad_enabled(False)

# %%
# load high level model
from circuits_benchmark.utils.iit import make_iit_hl_model
import circuits_benchmark.utils.iit.correspondence as correspondence
import iit.model_pairs as mp

def make_model_pair(benchmark_case):
    hl_model = benchmark_case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model, eval_mode=True)
    tracr_output = benchmark_case.get_tracr_output()
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(
            case=benchmark_case, tracr_output=tracr_output
        )
    model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)
    return model_pair

# %%
max_len = 100

# %%
from circuits_benchmark.utils.iit.dataset import get_unique_data


model_pair = make_model_pair(task)
unique_test_data = get_unique_data(task, max_len=max_len)

# %%
def collate_fn(batch):
    encoded_x = model_pair.hl_model.map_tracr_input_to_tl_input(list(zip(*batch))[0])
    return encoded_x

loader = torch.utils.data.DataLoader(unique_test_data, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)

# %% [markdown]
# ### Get the mean activations, norm and variance 

# %%

# %%
from utils.node_stats import get_node_stats, node_stats_to_df

cache_dict = get_node_stats(model_pair, loader)
node_norms = node_stats_to_df(cache_dict)

# %%
node_norms

# %%
import circuits_benchmark.commands.evaluation.iit.iit_eval as eval_node_effect

# model_pair = make_model_pair(task)
args = eval_node_effect.setup_args_parser(None, True)
max_len = 50
args.max_len = max_len
model_pair = make_model_pair(task)
node_effects, eval_metrics = eval_node_effect.get_node_effects(case=task, model_pair=model_pair, args=args, use_mean_cache=False)

# %%
node_effects

# %%
# combine node effects with node_norms
import pandas as pd
combined_df = pd.merge(node_effects, node_norms, left_on="node", right_on="name", how="inner")
combined_df.drop(columns=["name", "in_circuit"], inplace=True)
combined_df

# %%
import plotly.express as px

fig = px.scatter(combined_df, x="zero_ablate_effect", 
                 y="norm_cache", color="status",
                 error_y="norm_std",
                 # color map
                 color_discrete_map={
                    "in_circuit": "green",
                    "not_in_circuit": "orange",
                 },
                 labels={
                     "zero_ablate_effect": "Zero Ablation Effect",
                     "norm_cache": "Norm of Node Activation",
                     "status": "",
                     "resample_ablate_effect": "Resample Ablate Effect",
                 },
                 hover_data=["node", "resample_ablate_effect"],
                 # remove background grid and color
                 template="plotly_white",
                 # legend inside the plot
                 )

fig.show()
# save to file as pdf
fig.write_image(f"node_effect_norm_{task.get_index()}.pdf")

# %% [markdown]
# ### Do logit lens on all nodes in the model

# %% [markdown]
# 1. Decompose resid for each layer and get mlp logit lens
# 2. For head results I can do stack_head_results
# 
# Once I have the activations, I can compare it with (This is SCORE)
# 1. logit diffs for classification
# 2. only true regression output (maybe MSE with label or something)
# 
# Tuned lens: I just need to take the activations and train a linear layer b/w that and final layer act
# I have two choices:
# 1. train a map from hook points to pre unembed directly
# 2. train it on decomposed heads and resids
# 
# Then compute SCORE = unembed(LN(Linear(act))) vs logit diff/something
# 
# Pearson R coefficient:
# 1. Take the entire dataset, get SCORES for each prompt and calculate pearson R for all
# 
# Other experiments: 
# 1. I need to check if the mean is orthogonal/not to where we write. So I need the cosine similarity between acts. Combined with the fact that resample doesn't do shit, this makes sense. 
# 2. I can resample after multipyling the node with 1e-3, 1e-2 ... 10, 100 etc. and see it's effect. I can also do this after doing PCA on it, getting its subspace with max variation, and scaling that.

# %%
if model_pair.hl_model.is_categorical:
    # preprocess model for logit lens
    model.center_writing_weights(state_dict=model.state_dict())
    model.center_unembed(state_dict=model.state_dict())
    model.refactor_factored_attn_matrices(state_dict=model.state_dict())
try:
    model.fold_layer_norm(state_dict=model.state_dict())
except:
    print("No layer norm to fold")

# %%
import utils.logit_lens as logit_lens

logit_lens_results, labels = logit_lens.do_logit_lens(model_pair, loader)

# %%
logit_lens_results.keys()

# %%
from scipy import stats
import plotly.graph_objects as go
k = "0_mlp_out"

fig = go.Figure()

for i in range(logit_lens_results[k].shape[1]):
    y = labels[:, i].squeeze().detach().cpu().numpy()
    x = logit_lens_results[k][:, i].detach().cpu().numpy()
    pearson_corr = stats.pearsonr(x, y)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f"pos {i}, corr: {pearson_corr[0]:.2f}"))

fig.update_layout(title=f"Logit Lens Results for {k}", xaxis_title="True Logits", yaxis_title="Logit Lens Results")
fig.show()
    

# %%
pearson_corrs = {}
for k in logit_lens_results.keys():
    x = logit_lens_results[k].detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    for i in range(x.shape[1]): 
        pearson_corr = stats.pearsonr(x[:, i], y[:, i])
        if k not in pearson_corrs:
            pearson_corrs[k] = {}
        pearson_corrs[k][str(i)] = pearson_corr.correlation

pearson_corrs = pd.DataFrame(pearson_corrs)
px.imshow(pearson_corrs, 
          # set color map
            color_continuous_scale="Viridis",
            # set axis labels   
            labels=dict(x="Position", y="Layer/Head", color="Pearson Correlation"),
)
