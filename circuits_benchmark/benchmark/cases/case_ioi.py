from typing import Optional, Callable

import torch as t
from iit.model_pairs.base_model_pair import BaseModelPair
from iit.model_pairs.ioi_model_pair import IOI_ModelPair
from iit.tasks.ioi import ioi_cfg, IOI_HL, NAMES, IOIDatasetWrapper, make_corr_dict, suffixes, make_ll_edges
from iit.utils.correspondence import Correspondence
from iit.utils.nodes import LLNode
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookedRootModule

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_granularity import CircuitGranularity
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


class CaseIOI(BenchmarkCase):
    def __init__(self):
        self.ll_model = None
        self.hl_model = None

    def get_task_description(self) -> str:
        """Returns the task description for the benchmark case."""
        return "Indirect Object Identification (IOI) task."

    def get_clean_data(self,
                       min_samples: Optional[int] = 10,
                       max_samples: Optional[int] = 10,
                       seed: Optional[int] = 42,
                       unique_data: Optional[bool] = False) -> CaseDataset:
        ll_model = self.get_ll_model()
        ioi_dataset = IOIDatasetWrapper(
            num_samples=max_samples,
            tokenizer=ll_model.tokenizer,
            names=NAMES,
        )
        # We need to change IOIDatasetWrapper to inherit from CaseDataset if we want to remove the type ignore below
        return ioi_dataset  # type: ignore

    def get_max_seq_len(self) -> int:
        ioi_dataset = self.get_clean_data()
        x, *_ = ioi_dataset[0]
        return x.shape[0]

    def get_corrupted_data(self,
                           min_samples: Optional[int] = 10,
                           max_samples: Optional[int] = 10,
                           seed: Optional[int] = 43,
                           unique_data: Optional[bool] = False) -> CaseDataset:
        return self.get_clean_data(min_samples=min_samples, max_samples=max_samples, seed=seed, unique_data=unique_data)

    def get_validation_metric(
        self,
        ll_model: HookedTransformer,
        data: t.Tensor,
        *args, **kwargs
    ) -> Callable[[Tensor], Float[Tensor, ""]]:
        """Returns the validation metric for the benchmark case."""
        label_idx = IOI_ModelPair.get_label_idxs()
        clean_outputs = ll_model(data)

        def validation_metric(model_outputs):
            output_slice = model_outputs[label_idx.as_index]
            clean_outputs_slice = clean_outputs[label_idx.as_index]

            return t.nn.functional.kl_div(
                t.nn.functional.log_softmax(output_slice, dim=-1),
                t.nn.functional.softmax(clean_outputs_slice, dim=-1),
                reduction="batchmean",
            )

        return validation_metric

    def build_model_pair(
        self,
        training_args: dict | None = None,
        ll_model: HookedTransformer | None = None,
        hl_model: HookedRootModule | None = None,
        hl_ll_corr: Correspondence | None = None,
        *args, **kwargs
    ) -> BaseModelPair:
        if training_args is None:
            training_args = {}

        if ll_model is None:
            ll_model = self.get_ll_model()

        if hl_model is None:
            hl_model = self.get_hl_model()

        if hl_ll_corr is None:
            hl_ll_corr = self.get_correspondence()

        return IOI_ModelPair(
            ll_model=ll_model,
            hl_model=hl_model,
            corr=hl_ll_corr,
            training_args=training_args,
        )

    def get_ll_model_cfg(
        self,
        overwrite_cfg_dict: dict | None = None,
        eval: bool = False,
        *args, **kwargs
    ) -> HookedTransformerConfig:
        """Returns the configuration for the LL model for this benchmark case."""
        ll_cfg = HookedTransformer.from_pretrained(
            "gpt2"
        ).cfg.to_dict()
        ll_cfg.update(ioi_cfg)

        if overwrite_cfg_dict is not None:
            ll_cfg.update(overwrite_cfg_dict)

        if not eval:
            ll_cfg["init_weights"] = True
        else:
            ll_cfg["use_hook_mlp_in"] = True
            ll_cfg["use_attn_result"] = True
            ll_cfg["use_split_qkv_input"] = True

        return HookedTransformerConfig.from_dict(ll_cfg)

    def get_ll_model(
        self,
        device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
        overwrite_cfg_dict: dict | None = None,
        *args, **kwargs
    ) -> HookedTransformer:
        """Returns the untrained transformer_lens model for this case.
        In IIT terminology, this is the LL model before training."""
        if self.ll_model is not None:
            return self.ll_model

        ll_cfg = self.get_ll_model_cfg(overwrite_cfg_dict=overwrite_cfg_dict, *args, **kwargs)
        self.ll_model = HookedTransformer(ll_cfg).to(device)

        return self.ll_model

    def get_hl_model(
        self,
        device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
        *args, **kwargs
    ) -> HookedRootModule:
        """Builds the transformer_lens reference model for this case.
        In IIT terminology, this is the HL model."""
        if self.hl_model is not None:
            return self.hl_model

        ll_model = self.get_ll_model()
        names = t.tensor([ll_model.tokenizer.encode(name)[0] for name in NAMES]).to(device)
        self.hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out, names=names).to(device)

        return self.hl_model

    def get_correspondence(self,
                           include_mlp: bool = False,
                           eval: bool = False,
                           use_pos_embed: bool = False,
                           *args, **kwargs) -> Correspondence:
        """Returns the correspondence between the reference and the benchmark model."""
        corr_dict = make_corr_dict(include_mlp=include_mlp, eval=eval, use_pos_embed=use_pos_embed)
        return Correspondence.make_corr_from_dict(corr_dict, suffixes=suffixes)

    def is_categorical(self) -> bool:
        """Returns whether the benchmark case is categorical."""
        return True

    def get_ll_gt_circuit(self,
                          granularity: CircuitGranularity = "acdc_hooks",
                          corr: Correspondence | None = None,
                          *args, **kwargs) -> Circuit:
        """Returns the ground truth circuit for the LL model."""
        if corr is None:
            corr = self.get_correspondence()

        def make_circuit_node(ll_node: LLNode):
            if 'attn' in ll_node.name:
                index = ll_node.index
                head = index.as_index[2]
                node_name = ll_node.name.replace("hook_z", "hook_result")
                return CircuitNode(node_name, head)
            if "mlp" in ll_node.name:
                node_name = ll_node.name.replace("mlp.hook_post", "hook_mlp_out")
                return CircuitNode(node_name, None)
            return CircuitNode(ll_node.name, None)

        gt_circuit = Circuit()
        for edge in make_ll_edges(corr):
            circuit_node_from = make_circuit_node(edge[0])
            circuit_node_to = make_circuit_node(edge[1])
            gt_circuit.add_edge(circuit_node_from, circuit_node_to)

        return gt_circuit

    def get_hl_gt_circuit(self, granularity: CircuitGranularity = "acdc_hooks", *args, **kwargs) -> Circuit:
        """Returns the ground truth circuit for the HL model."""
        raise NotImplementedError()
