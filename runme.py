import os
import pickle
# from iit_utils.acdc_eval import run_acdc_eval
task=3
weights = ["110", "510", "100" ,"910"]
ibs = ["-b 1.0 -i 1.0 -s 0.0", "-b 1.0 -i 1.0 -s 0.4", "-b 1.0 -i 0.0 -s 0.0", "-b 1.0 -i 1.0 -s 0.8"]
wandb = True
# weights = [weights[1], weights[2]]
# ibs = [ibs[1], ibs[2]]
############### train on task
command = """python iit_train.py -t {} {}"""
for i in (range(len(ibs))):
    command_to_run = command.format(task, ibs[i])
    print("Running", command_to_run)
    os.system(command_to_run)

################ evaluate on task
command = """python iit_eval.py -t {} -w {}"""
for weight in weights:
    command_to_run = command.format(task, weight)
    print("Running", command_to_run)
    os.system(command_to_run)

################# run acdc
thresholds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 
              0.025, 0.05, 0.1, 0.5, 0.8, 1.0, 10.0]
command = """python acdc.py -c {} -w {} -t {}"""
results = {}
for weight in weights:
    for threshold in thresholds:
        command_to_run = command.format(task, weight, threshold)
        command_to_run += " -wandb" if wandb else ""
        print("=====================================================================================================")
        print("Running", command_to_run)
        print("=====================================================================================================")
        file = f"results/acdc_{task}/weight_{weight}/threshold_{threshold}/result.pkl"
        # if os.path.exists(file):
        #     result = pickle.load(open(file, "rb"))
        # else:
        #     os.system(command_to_run)
        #     if not os.path.exists(file):
        #         print("ERROR: ", file, "not found")
        #         continue
        #     result = pickle.load(open(file, "rb"))
        os.system(command_to_run)
        result = pickle.load(open(file, "rb"))
        results[(weight, threshold)] = result
        
pickle.dump(results, open(f"results/acdc_{task}/results.pkl", "wb"))