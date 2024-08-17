from utils.sweep import Sweep
import pickle
from tqdm import tqdm
import os

# %%
# def get_sweep_results_for_strict_weight(task_idx, strict_weights):
#     strict_weights_sweep_expt = Sweep(
#         task_idx, "strict_weight", strict_weights, control_params=["--epochs", "10"]
#     )
#     strict_weights_sweep_expt.run_sweep()
#     return strict_weights_sweep_expt.sweep_results


# tasks = ['3', '4', '8', '21', '24']
# strict_weights = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# strict_weights_sweep_expt = {}

# for task in tqdm(tasks):
#     strict_weights_sweep_expt[task] = get_sweep_results_for_strict_weight(task, strict_weights)


# pickle.dump(strict_weights_sweep_expt, open('results/strict_weights_sweep_expt.pkl', 'wb'))
# %%
# def get_sweep_results_for_seed(task_idx, seeds):
#     seeds_sweep_expt = Sweep(task_idx, "seed", seeds, control_params=["--epochs", "10"])
#     seeds_sweep_expt.run_sweep()
#     return seeds_sweep_expt.sweep_results


# tasks = ["3", "4", "8", "21", "24"]
# seeds = range(10)

# seeds_sweep_expt = {}

# for task in tqdm(tasks):
#     seeds_sweep_expt[task] = get_sweep_results_for_seed(task, seeds)

# pickle.dump(seeds_sweep_expt, open("results/seeds_sweep_expt.pkl", "wb"))


# %%
def get_sweep_results_for_both_seeds_and_strict_weight(task_idx, seeds, strict_weights):
    sweep_results = {}
    for strict_weight in strict_weights:
        seeds_sweep_expt = Sweep(
            task_idx,
            "seed",
            seeds,
            control_params=["--epochs", "10", "--strict_weight", str(strict_weight)],
        )
        seeds_sweep_expt.run_sweep(len(seeds))
        seed_results = seeds_sweep_expt.sweep_results
        sweep_results[strict_weight] = {seeds[i]: seed_results[i] for i in range(len(seeds))}
    return sweep_results


tasks = ["3", "4"]
seed_strict_weight_sweep_expt = (
    pickle.load(open("results/seed_strict_weight_sweep_expt.pkl", "rb"))
    if os.path.exists("results/seed_strict_weight_sweep_expt.pkl")
    else {}
)
strict_weights = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
seeds = range(5)
for task in tqdm(tasks):
    seed_strict_weight_sweep_expt[task] = get_sweep_results_for_both_seeds_and_strict_weight(
        task, seeds, strict_weights
    )
    pickle.dump(
        seed_strict_weight_sweep_expt, open("results/seed_strict_weight_sweep_expt.pkl", "wb")
    )
