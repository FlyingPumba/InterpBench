import os

working = [11, 13, 18, 19, 20, 21, 26, 29, 3, 33, 34, 35, 36, 37, 4, 8]
for i in working:
    os.system(f"python main.py eval gt_node_realism -i {i} --mean -w best --load-from-wandb --max-len 50000")