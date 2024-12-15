from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from jaxtyping import Float
import torch
import os
import shutil
from argparse import Namespace
from copy import deepcopy
from torch.utils.data import Dataset
from transformers import TrainingArguments, Seq2SeqTrainingArguments
import pickle

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.circuit.circuit_eval import evaluate_hypothesis_circuit, CircuitEvalResult
from circuits_benchmark.utils.edge_pruning.prune.erazr_reverse import ErazrTrainer, get_optimizers
from circuits_benchmark.utils.edge_pruning.modeling.modeling_erazr import ErazrConfig, ErazrModelForSequenceTransformation
from circuits_benchmark.utils.edge_pruning.prune.erazr_reverse import DataCollatorReverse
from circuits_benchmark.commands.common_args import add_common_args, add_evaluation_common_ags

@dataclass
class EdgePruningConfig:
    output_dir: str
    start_edge_sparsity: float = 0.0
    target_edge_sparsity: float = 0.92
    start_node_sparsity: float = 0.0 
    target_node_sparsity: float = 0.16
    num_sparsity_warmup_steps: int = 250
    edge_learning_rate: float = 1e-2
    node_learning_rate: float = 1.0  # Original: 1.0
    reg_edge_learning_rate: float = 1e-2
    reg_node_learning_rate: float = 1.0  # Original: 1.0
    warmup_type: str = "linear"
    disable_node_loss: bool = False
    edge_threshold: float = 0.005
    node_threshold: float = 0.005
    max_steps: int = 2000
    save_steps: int = 500
    logging_steps: int = 100
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    device: str = "cuda"
    data_size: int = 1000
    use_pos_embed: bool = False
    same_size: bool = False
    using_wandb: bool = False
    wandb_project_name: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_args(cls, args: Namespace) -> 'EdgePruningConfig':
        return cls(
            output_dir=args.output_dir,
            start_edge_sparsity=args.start_edge_sparsity,
            target_edge_sparsity=args.target_edge_sparsity,
            start_node_sparsity=args.start_node_sparsity,
            target_node_sparsity=args.target_node_sparsity,
            num_sparsity_warmup_steps=args.num_sparsity_warmup_steps,
            edge_learning_rate=args.edge_learning_rate,
            node_learning_rate=args.node_learning_rate,
            reg_edge_learning_rate=args.reg_edge_learning_rate,
            reg_node_learning_rate=args.reg_node_learning_rate,
            warmup_type=args.warmup_type,
            disable_node_loss=args.disable_node_loss,
            edge_threshold=args.edge_threshold,
            node_threshold=args.node_threshold,
            max_steps=args.max_steps,
            device=args.device,
            data_size=args.data_size,
            use_pos_embed=args.use_pos_embed,
            same_size=args.same_size,
            using_wandb=args.using_wandb,
            wandb_project_name=args.wandb_project_name,
            wandb_run_name=args.wandb_run_name
        )


class DataCollatorTracr:
    def __init__(self, tokenizer=None, length=None):
        self.tokenizer = tokenizer
        self.length = length

    def __call__(self, examples):
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []
        
        for i, example in enumerate(examples):
            input_ids_all.append(example["seq"])
            corr_input_ids_all.append(example["corr_seq"])
            labels_all.append(example["target"])
        
        if not input_ids_all:
            raise ValueError("No valid examples were processed")
            
        return {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
        }


class EdgePruningDataset(Dataset):
    def __init__(self, clean_inputs, clean_outputs, corrupted_inputs):
        self.clean_inputs = clean_inputs
        self.clean_outputs = clean_outputs
        self.corrupted_inputs = corrupted_inputs

    def __len__(self):
        return self.clean_inputs.shape[0]

    def __getitem__(self, idx):
        item = {
            "seq": self.clean_inputs[idx],
            "target": self.clean_outputs[idx],
            "corr_seq": self.corrupted_inputs[idx],
        }
        return item


class EdgePruningRunner:
    def __init__(
        self,
        case: BenchmarkCase,
        config: EdgePruningConfig | None = None,
        args: Namespace | None = None
    ):
        self.case = case
        self.config = config
        self.args = deepcopy(args)

        if self.config is None:
            self.config = EdgePruningConfig.from_args(args)

        assert self.config is not None

    def run_using_model_loader(self, ll_model_loader: LLModelLoader) -> Tuple[Circuit, CircuitEvalResult]:
        clean_dirname = self.prepare_output_dir(ll_model_loader)
        
        print(f"Running Edge Pruning evaluation for case {self.case.get_name()} ({str(ll_model_loader)})")
        print(f"Output directory: {clean_dirname}")

        hl_ll_corr, ll_model = ll_model_loader.load_ll_model_and_correspondence(
            device=self.config.device,
            output_dir=self.config.output_dir,
            same_size=self.config.same_size,
            use_pos_embed=self.config.use_pos_embed
        )

        # Get datasets
        clean_dataset = self.case.get_clean_data(max_samples=self.config.data_size)
        corrupted_dataset = self.case.get_corrupted_data(max_samples=self.config.data_size)

        # Run edge pruning
        erazr_circuit = self.run(
            ll_model,
            clean_dataset.get_inputs(),
            clean_dataset.get_targets(),
            corrupted_dataset.get_inputs(),
            corrupted_dataset.get_targets(),
        )

        print("hl_ll_corr:", hl_ll_corr)
        hl_ll_corr.save(f"{clean_dirname}/hl_ll_corr.pkl")

        print("Calculating circuit evaluation metrics")
        result = evaluate_hypothesis_circuit(
            erazr_circuit,
            ll_model,
            hl_ll_corr,
            self.case,
        )

        # Save results
        with open(f"{clean_dirname}/result.txt", "w") as f:
            f.write(str(result))

        with open(f"{clean_dirname}/result.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"Saved result to {clean_dirname}/result.txt and {clean_dirname}/result.pkl")

        # if self.config.using_wandb:
        #     import wandb
        #     wandb.init(
        #         project="circuit_discovery",
        #         group=f"edge_pruning_{self.case.get_name()}_{ll_model_loader.get_output_suffix()}",
        #         name=f"sparsity_{self.config.target_edge_sparsity}",
        #     )
        #     wandb.save(f"{clean_dirname}/*", base_path=self.config.output_dir)

        return erazr_circuit, result

    @staticmethod
    def setup_subparser(subparsers):
        parser = subparsers.add_parser("edge_pruning")
        EdgePruningRunner.add_args_to_parser(parser)

    @staticmethod
    def add_args_to_parser(parser):
        add_common_args(parser)
        add_evaluation_common_ags(parser)

        parser.add_argument("--start-edge-sparsity", type=float, default=0.0)
        parser.add_argument("--target-edge-sparsity", type=float, default=0.92)
        parser.add_argument("--start-node-sparsity", type=float, default=0.0)
        parser.add_argument("--target-node-sparsity", type=float, default=0.16)
        parser.add_argument("--num-sparsity-warmup-steps", type=int, default=1000)
        parser.add_argument("--edge-learning-rate", type=float, default=1e-2)
        parser.add_argument("--node-learning-rate", type=float, default=1.0)
        parser.add_argument("--reg-edge-learning-rate", type=float, default=1e-2)
        parser.add_argument("--reg-node-learning-rate", type=float, default=1.0)
        parser.add_argument("--warmup-type", type=str, default="linear")
        parser.add_argument("--disable-node-loss", action="store_true")
        parser.add_argument("--edge-threshold", type=float, default=0.005)
        parser.add_argument("--node-threshold", type=float, default=0.005)
        parser.add_argument("--max-steps", type=int, default=2000)
        parser.add_argument("--data-size", type=int, default=1000)
        parser.add_argument("--use-pos-embed", action="store_true")
        parser.add_argument("-wandb", "--using_wandb", action="store_true")
        parser.add_argument("--wandb-project-name", type=str, default=None)
        parser.add_argument("--wandb-run-name", type=str, default=None)

    def run(
        self,
        model: torch.nn.Module,
        clean_inputs: Float[torch.Tensor, "batch seq_len"],
        clean_outputs: Float[torch.Tensor, "batch seq_len 1"],
        corrupted_inputs: Float[torch.Tensor, "batch seq_len"],
        corrupted_outputs: Float[torch.Tensor, "batch seq_len 1"],
    ) -> Circuit:
        # assert we have data
        assert clean_inputs.shape[0] > 0
        assert clean_outputs.shape[0] > 0
        assert corrupted_inputs.shape[0] > 0
        assert corrupted_outputs.shape[0] > 0

        # Initialize ERAZR model with appropriate config
        config = ErazrConfig(
            model_size=model.cfg.d_model,
            num_layers=model.cfg.n_layers,
            num_heads=model.cfg.n_heads,
            key_size=model.cfg.d_head,
            value_size=model.cfg.d_head,
            mlp_hidden_size=model.cfg.d_mlp,
            vocab_size=model.cfg.d_vocab,
            max_position_embeddings=model.cfg.n_ctx,
        )
        erazr_model = ErazrModelForSequenceTransformation(config)

        # Extract weights in the format expected by load_everything
        embeds_weights = model.embed.W_E
        pos_embeds_weights = model.pos_embed.W_pos
        unembedding_mtx = model.unembed.W_U
        
        block_weights = []
        for layer in model.blocks:
            w_q = layer.attn.W_Q
            w_k = layer.attn.W_K  
            w_v = layer.attn.W_V
            w_o = layer.attn.W_O
            b_q = layer.attn.b_Q if hasattr(layer.attn, 'b_Q') else None
            b_k = layer.attn.b_K if hasattr(layer.attn, 'b_K') else None
            b_v = layer.attn.b_V if hasattr(layer.attn, 'b_V') else None
            b_o = layer.attn.b_O if hasattr(layer.attn, 'b_O') else None
            
            up_proj = layer.mlp.W_in
            down_proj = layer.mlp.W_out
            b_up = layer.mlp.b_in if hasattr(layer.mlp, 'b_in') else None
            b_down = layer.mlp.b_out if hasattr(layer.mlp, 'b_out') else None
            
            block_weights.append([w_q, w_k, w_v, w_o, up_proj, down_proj, 
                                b_q, b_k, b_v, b_o, b_up, b_down])

        # Load weights using load_everything
        erazr_model.load_everything(embeds_weights, pos_embeds_weights, unembedding_mtx, block_weights)

        tracr_model = ErazrModelForSequenceTransformation(config)
        tracr_model.load_everything(embeds_weights, pos_embeds_weights, unembedding_mtx, block_weights)

        # Move model to correct device
        erazr_model = erazr_model.to(model.cfg.device)
        tracr_model = tracr_model.to(model.cfg.device)

        # Set thresholds
        erazr_model.set_edge_threshold_for_deterministic(self.config.edge_threshold)
        erazr_model.set_node_threshold_for_deterministic(self.config.node_threshold)

        # Prepare training arguments
        wandb_run_name = self.config.wandb_run_name
        if wandb_run_name is None:
            wandb_run_name = f"edge_pruning_{self.case.get_name()}"

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            do_train=True,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.edge_learning_rate,
            warmup_steps=self.config.num_sparsity_warmup_steps // 2,
            remove_unused_columns=False,
            generation_config=None,
            report_to=["wandb"] if self.config.using_wandb else [],
            run_name=wandb_run_name,
        )
        # Set WANDB_PROJECT environment variable
        if self.config.using_wandb and self.config.wandb_project_name is not None:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project_name

        # Create data collator
        collator = DataCollatorTracr()

        # Build train_dataset as a list of dicts with keys "seq", "target", "corr_seq"
        train_dataset = EdgePruningDataset(clean_inputs, clean_outputs, corrupted_inputs)

        # Get optimizers
        optimizers = get_optimizers(
            model=erazr_model,
            edges_lr=self.config.edge_learning_rate,
            nodes_lr=self.config.node_learning_rate,
            reg_edges_lr=self.config.reg_edge_learning_rate,
            reg_nodes_lr=self.config.reg_node_learning_rate,
            num_training_steps=self.config.max_steps,
            warmup_steps=self.config.num_sparsity_warmup_steps // 2,
        )

        # Initialize trainer
        trainer = ErazrTrainer(
            model=erazr_model,
            tracr_model=tracr_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            optimizers=optimizers,  # Pass optimizers separately
            # Edge pruning specific arguments
            start_edge_sparsity=self.config.start_edge_sparsity,
            target_edge_sparsity=self.config.target_edge_sparsity,
            start_node_sparsity=self.config.start_node_sparsity,
            target_node_sparsity=self.config.target_node_sparsity,
            num_sparsity_warmup_steps=self.config.num_sparsity_warmup_steps,
            warmup_type=self.config.warmup_type,
            disable_node_loss=self.config.disable_node_loss,
        )

        # Train
        trainer.train()

        # Get final circuit
        erazr_model.eval()
        with torch.no_grad():
            edges = erazr_model.get_edges()
            
        return Circuit(edges)

    def prepare_output_dir(self, ll_model_loader: LLModelLoader) -> str:
        output_suffix = (
            f"{ll_model_loader.get_output_suffix()}/"
            f"edge_sparsity_{self.config.target_edge_sparsity}_"
            f"node_sparsity_{self.config.target_node_sparsity}"
        )
        clean_dirname = os.path.join(
            self.config.output_dir,
            "edge_pruning",
            self.case.get_name(),
            output_suffix
        )

        # Remove everything in the directory
        if os.path.exists(clean_dirname):
            shutil.rmtree(clean_dirname)

        # Create directory
        os.makedirs(clean_dirname, exist_ok=True)

        return clean_dirname