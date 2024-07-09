import os
import argparse
from multiprocessing import Process, Semaphore
from circuits_benchmark.utils.iit.best_weights import get_best_weight
import wandb

working = [11, 13, 18, 19, 20, 21, 26, 29, 3, 33, 34, 35, 36, 37, 4, 8]
thresholds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 10.0, 20.0, 50.0, 100.0]
regression = [3, 4]
command_templates_with_wandb = {
    "acdc": """python main.py eval iit_acdc -i {} -w {} -t {} -wandb --load-from-wandb --abs-value-threshold""",
    "eap": """python main.py eval iit_eap -i {} -w {} --threshold {} --load-from-wandb -wandb""",
    "eap_integrated_grad": """python main.py eval iit_eap -i {} -w {} --threshold {} --integrated-grad-steps 5 --load-from-wandb -wandb""",

    # group: case_weight 
    # name: lambda_reg
    "node_sp": """python main.py run sp -i {} -w {} --lambda-reg {} --metric {} --torch-num-threads 4 --device cuda  --epochs 2000 --load-from-wandb --using-wandb --wandb-project circuit_discovery --wandb-group node_sp_{}_{} --wandb-run-name {}""",
    "edge_sp": """python main.py run sp -i {} -w {} --lambda-reg {} --metric {} --torch-num-threads 4 --device cuda  --epochs 3000 --load-from-wandb --using-wandb --wandb-project circuit_discovery --wandb-group edge_sp_{}_{} --wandb-run-name {} --edgewise""",
}

command_templates_without_wandb = {
    "acdc": """python main.py eval iit_acdc -i {} -w {} -t {} --load-from-wandb --abs-value-threshold""",
    "eap": """python main.py eval iit_eap -i {} -w {} --threshold {} --load-from-wandb""",
    "eap_integrated_grad": """python main.py eval iit_eap -i {} -w {} --threshold {} --integrated-grad-steps 5 --load-from-wandb""",
    "node_sp": """python main.py run sp -i {} -w {} --lambda-reg {} --metric {} --torch-num-threads 4 --device cuda  --epochs 2000 --load-from-wandb --wandb-project circuit_discovery --wandb-group node_sp_{}_{} --wandb-run-name {}""",
    "edge_sp": """python main.py run sp -i {} -w {} --lambda-reg {} --metric {} --torch-num-threads 4 --device cuda  --epochs 3000 --load-from-wandb --wandb-project circuit_discovery --wandb-group edge_sp_{}_{} --wandb-run-name {} --edgewise""",
}

def delete_wandb_runs(case, algorithm, weight):
    api = wandb.Api()
    runs = api.runs("circuit_discovery")
    for run in runs:
        if algorithm in run.group and str(case) in run.group and weight in run.group:
            run.delete()
            print(f"Deleted run {run.group}: {run.name}")

def run_circuit_discovery(i, algorithm, weight, threshold, use_wandb=True):
    command_templates = command_templates_with_wandb if use_wandb else command_templates_without_wandb
    if algorithm in ["acdc", "eap", "eap_integrated_grad"]:
        command = command_templates[algorithm].format(i, weight, threshold)
    elif algorithm in ["node_sp", "edge_sp"]:
        metric = "l2" if i in regression else "kl"
        command = command_templates[algorithm].format(i, weight, threshold, metric, i, weight, threshold)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")
    print(command)
    os.system(command)

def run_all_thresholds_for_case(i, algorithms, weight, use_wandb=True, max_processes=4):
    if weight == "best":
        weight = get_best_weight(i, individual=False)
    processes = []
    sema = Semaphore(max_processes)
    for threshold in thresholds:
        for algorithm in algorithms:
            if use_wandb:
                delete_wandb_runs(i, algorithm, weight)
            sema.acquire()
            p = Process(target=run_circuit_discovery, args=(i, algorithm, weight, threshold, use_wandb))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc", action="store_true", help="Run ACDC")
    parser.add_argument("--eap", action="store_true", help="Run EAP")
    parser.add_argument("--eap-integrated-grad", action="store_true", help="Run EAP with integrated gradients")
    parser.add_argument("--node-sp", action="store_true", help="Run Node SP")
    parser.add_argument("--edge-sp", action="store_true", help="Run Edge SP")
    parser.add_argument("--all", action="store_true", help="Run all algorithms")
    parser.add_argument("-i", "--case", type=int, help="Case number", default=None)
    parser.add_argument("-w", "--weight", type=str, help="Weights", default="best")
    parser.add_argument("--all-cases", action="store_true", help="Run all cases")
    parser.add_argument("-wandb", "--use-wandb", action="store_true", help="Use wandb")
    parser.add_argument("--max-processes", type=int, help="Max number of processes", default=4)
    args = parser.parse_args()

    assert (
        args.acdc or args.eap or args.eap_integrated_grad or args.node_sp or args.edge_sp or args.all
    ), "Please specify at least one algorithm to run"
    assert args.case is not None or args.all_cases, "Please specify a case number or use --all-cases"

    algorithms = []
    if args.acdc or args.all:
        algorithms.append("acdc")
    if args.eap or args.all:
        algorithms.append("eap")
    if args.eap_integrated_grad or args.all:
        algorithms.append("eap_integrated_grad")
    if args.node_sp or args.all:
        algorithms.append("node_sp")
    if args.edge_sp or args.all:
        algorithms.append("edge_sp")

    if args.all_cases:
        max_processes = args.max_processes
        sema = Semaphore(max_processes)
        # run parallel processes for each case
        processes = []
        for i in working:
            sema.acquire()
            p = Process(target=run_all_thresholds_for_case, args=(i, algorithms, args.weight, args.use_wandb))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        run_all_thresholds_for_case(args.case, algorithms, args.weight, args.use_wandb, args.max_processes)
    
    print("Done!")