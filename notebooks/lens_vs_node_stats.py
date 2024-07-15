# %%
from interp_utils.node_stats.all_stats import make_all_stats
from circuits_benchmark.utils.get_cases import get_names_of_working_cases
import pandas as pd
from typing import Literal
import re
from interp_utils.common import append_row
from interp_utils.node_stats.plotting import make_scatter_plot
from functools import partial
import matplotlib.pyplot as plt
import wandb

# %%

make = False
csv_file = "interp_results/siit_node_stats_all_cases.csv"
if make:
    cases = [ case for case in get_names_of_working_cases() if 'ioi' not in case ]

    siit_stats = make_all_stats(
        cases=cases,
        only_siit=True,
    )

    siit_stats.to_csv(csv_file, index=False)
else:
    siit_stats = pd.read_csv(csv_file)
    cases = siit_stats['run'].unique()


print(cases)

# %%

lens: Literal['logit_lens', 'tuned_lens'] = 'logit_lens'
lens_stats = {}

for case in cases:
    pearsons_for_case = pd.read_csv(f"interp_results/{case}/{lens}/combined_pearson.csv")
    p_value_for_case = pd.read_csv(f"interp_results/{case}/{lens}/combined_p_values.csv")
    lens_stats[case] = {
        'pearsons': pearsons_for_case,
        'p_values': p_value_for_case,
    }

print(lens_stats.keys())

# %%


lens_node_templates = {
    "mlp" : """{layer}_mlp_out""",
    "attn": """L{layer}H{head}""",
    "in_circuit": "(IC)"
}

siit_node_templates = {
    "mlp": """blocks.{layer}.mlp.hook_post""",
    "attn": """blocks.{layer}.attn.hook_result, head  {head}""",
}

def convert_lens_node_name_to_siit_node_name(lens_node_name: str) -> str:
    # check if lens node matches the mlp template
    lens_matches_mlp = re.match(r"(\d)_mlp_out", lens_node_name)
    # check if lens node matches the attn template
    lens_matches_attn = re.match(r"L(\d)H(\d)", lens_node_name)

    if lens_matches_mlp:
        layer = lens_node_name.split("_")[0]
        return siit_node_templates["mlp"].format(layer=layer)
    
    elif lens_matches_attn:
        layer = lens_node_name[1]
        head = lens_node_name[3]
        return siit_node_templates["attn"].format(layer=layer, head=head)
    else:
        return None
    
assert convert_lens_node_name_to_siit_node_name("3_mlp_out") == "blocks.3.mlp.hook_post"
assert convert_lens_node_name_to_siit_node_name("L3H2") == "blocks.3.attn.hook_result, head  2"
assert convert_lens_node_name_to_siit_node_name("L0H1(IC)") == "blocks.0.attn.hook_result, head  1"

# %%

def make_lens_stats_df_for_case(
    case: int | str,
    lens_stats: dict,
) -> pd.DataFrame:
    pearson = lens_stats[case]['pearsons']
    p_value = lens_stats[case]['p_values']
    
    lens_stats_df = pd.DataFrame(
        columns=["node", "pearson", "p_value"]
    )

    for node in pearson.columns:
        siit_node_name = convert_lens_node_name_to_siit_node_name(node)
        if siit_node_name is None:
            continue

        pearson_val = pearson[node].values[0]
        p_value_val = p_value[node].values[0]

        lens_stats_df_entry = {
            "node": siit_node_name,
            "pearson": pearson_val,
            "p_value": p_value_val
        }

        lens_stats_df = append_row(lens_stats_df, pd.Series(lens_stats_df_entry))
    
    return lens_stats_df

for i, j in zip(sorted(siit_stats[siit_stats["run"] == 3]["node"].values), sorted(make_lens_stats_df_for_case(3, lens_stats)["node"].values)):
    assert i == j, f"{i} != {j}"

# %%
def merge_siit_and_lens_df_for_case(
    case: int | str,
    siit_stats: pd.DataFrame,
    lens_stats: dict,
) -> pd.DataFrame:

    siit_stats_for_case = siit_stats[siit_stats['run'] == case]

    # make logit_lens_stats_df for case
    lens_stats_df = make_lens_stats_df_for_case(
        case=case,
        lens_stats=lens_stats,
    )
    # merge with siit stats for case
    merged_df = pd.merge(
        siit_stats_for_case,
        lens_stats_df,
        on="node",
        how="inner",
    )
    return merged_df

# %%
combined_siit_stats_list_by_case = []

for k, v in lens_stats.items():
    combined_df = merge_siit_and_lens_df_for_case(
        case=k,
        siit_stats=siit_stats,
        lens_stats=lens_stats,
    )
    combined_siit_stats_list_by_case.append(combined_df)

for case in combined_siit_stats_list_by_case:
    # check if the number of rows in the combined df is the same as the number of rows in the siit stats df
    assert len(case) == len(siit_stats[siit_stats['run'] == case['run'].values[0]]), f"{len(case)} != {len(siit_stats[siit_stats['run'] == case['run'].values[0]])}"

# %%
# append all dfs in the list
combined_siit_stats = pd.concat(combined_siit_stats_list_by_case)
combined_siit_stats = combined_siit_stats.loc[:, ~combined_siit_stats.columns.str.contains('^Unnamed')]
combined_siit_stats.to_csv(f"interp_results/siit_{lens}_combined.csv", index=False)
combined_siit_stats

# %%
# normalise node norm and std per case
def normalise_node_norm_and_std(
    df: pd.DataFrame,
    norm_col: str,
    std_col: str,
) -> pd.DataFrame:
    new_df = pd.DataFrame(columns=df.columns)
    for case in df['run'].unique():
        case_df = df[df['run'] == case]
        case_df.loc[:, norm_col] = (case_df[norm_col] - case_df[norm_col].mean()) / case_df[norm_col].std()
        case_df.loc[:, std_col] = (case_df[std_col] - case_df[std_col].mean()) / case_df[std_col].std()
        new_df = pd.concat([new_df,
                            case_df])
    return new_df

normalised_siit_stats = normalise_node_norm_and_std(
    df=combined_siit_stats,
    norm_col="norm_cache",
    std_col="norm_std",
)

# %%


def make_alpha_col_for_id(row, smaller_is_better=True, row_id = "p_value", row_normalizer = 0.05):
    if smaller_is_better:
        if row[row_id] > 0.05:
            return 0.2
        return  0.7 - (row[row_id] / row_normalizer) * 0.5
    else:
        if row[row_id] < 0.05:
            return 0.2
        return  0.7 * (row[row_id] / row_normalizer)


fig = make_scatter_plot(
    df=normalised_siit_stats[normalised_siit_stats["zero_ablate_effect"] != 0],
    x="pearson",
    y="zero_ablate_effect",
    make_alpha_fn=make_alpha_col_for_id,
)

fig.write_html(f"interp_results/pearson_plots/siit_{lens}_pearson_vs_zero_ablate_effect.html")
# fig

# %%


def make_size_col(row, row_id, row_normalizer):
    return 18 + (row[row_id] / row_normalizer) * 5

def make_alpha_col(row):
    if row["p_value"] > 0.05:
        return 0.0
    return 0.2 + (row["zero_ablate_effect"]**0.5) * 0.8

fig = make_scatter_plot(
    df=normalised_siit_stats[normalised_siit_stats["zero_ablate_effect"] != 0],
    x="pearson",
    y="norm_cache",
    y_err=None,
    make_alpha_fn=lambda row: make_alpha_col(row),
    make_size_fn=partial(make_size_col, row_id="norm_std", row_normalizer=1),
    # ylog=True,
)

# add a red horizontal line at 0
fig.add_hline(y=0, line_dash="dash", line_color="red", name="mean node norm line", showlegend=True)
fig.update_layout(
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        # font size
        font=dict(size=15),
    ),
)

fig.write_html(f"interp_results/pearson_plots/siit_{lens}_pearson_vs_norm_cache.html")

# %%
pearson_df = pd.DataFrame({
    "pearson": [pearson_vals for pearson_vals in normalised_siit_stats["pearson"].values],
    "p_value": normalised_siit_stats["p_value"].values,
    "ones": [1 if status == "in_circuit" else 0 for status in normalised_siit_stats["status"].values],
    "status": normalised_siit_stats["status"].values
})
fig = pearson_df.boxplot(column="pearson", by="status")


plt.suptitle("")
plt.title("")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Node status", fontsize=16)
plt.ylabel("Pearson correlation", fontsize=16)
fig.get_figure().set_size_inches(6, 5)

plt.show()

# save the figure
fig.get_figure().savefig(f"interp_results/pearson_plots/siit_{lens}_pearson_boxplot.png")

# %%


wandb.init(project="siit_node_stats")

wandb.log({
    f"siit_{lens}_pearson_vs_zero_ablate_effect".replace("_", " "): 
    wandb.Html(f"interp_results/pearson_plots/siit_{lens}_pearson_vs_zero_ablate_effect.html"),
    f"siit_{lens}_pearson_vs_norm_cache".replace("_", " "):
    wandb.Html(f"interp_results/pearson_plots/siit_{lens}_pearson_vs_norm_cache.html"),
    f"siit_{lens}_pearson_boxplot".replace("_", " "):
    wandb.Image(f"interp_results/pearson_plots/siit_{lens}_pearson_boxplot.png"),
})
wandb.finish()

# %%



