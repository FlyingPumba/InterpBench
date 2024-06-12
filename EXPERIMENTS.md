The following commands can be used to replicate the experiments presented in the paper "InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques".

For training the SIIT models on Tracr tasks (where `-i 3` is the index of the task), and training the IOI model:
- `python main.py train iit -i 3 --epochs 500 --model-pair strict -iit 1 -s 0.4 -b 1`
- `python main.py train ioi --include-mlp --next-token --epochs 10 --save-to-wandb`

For evaluating the effect of nodes and the accuracy after ablating everything but ground truth circuit:
- `python main.py eval iit -i 3 --categorical-metric kl_div -w best`
- `python main.py eval ioi --next-token --include-mlp`
- `python main.py eval gt_node_realism -i 3 --mean -w best --relative 1`


For running the performance evaluation of circuit discovery techniques:
- `python main.py run sp --loss-type l2 -i 3 --torch-num-threads 0 --device cpu --epochs 500 --atol 0.1`
- `python main.py run sp --loss-type l2 -i 3 --torch-num-threads 0 --device cpu --epochs 500 --atol 0.1 --edgewise`
- `python main.py eval iit_acdc -i 2 -w 100 -t 0.0 --load-from-wandb`

For running the experiments on realism:
- `python main.py eval node_realism -i 3 --mean --relative 1 --algorithm acdc --tracr -t 0`
- `python main.py eval node_realism -i 3 --mean --relative 1 --algorithm node_sp -t 0`
- `python main.py eval node_realism -i 3 --mean --relative 1 --algorithm edge_sp -t 0`
- `python main.py eval ioi_acdc --data-size 10 --max-num-epochs 1 threshold 1000.0 --next-token --include-mlp`