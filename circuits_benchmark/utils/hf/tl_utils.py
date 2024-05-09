from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    GPT2Config,
    GPT2Model,
)
from pyvene import IntervenableModel, IntervenableConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pyvene.models.constants import *
from .tl_wrapper_model import TLModel, output_type
from pyvene.models.intervenable_modelcard import (
    type_to_module_mapping,
    type_to_dimension_mapping,
)
from .tl_config import tl_to_module_mapping, tl_to_dimension_mapping


def make_gpt2_tl_model(device="cpu", output_type=output_type.base_model_output):
    model = HookedTransformer.from_pretrained("gpt2")
    model.to(device)
    ll_cfg = model.cfg.to_dict()
    my_config = GPT2Config(
        vocab_size=ll_cfg["d_vocab"],
        n_positions=ll_cfg["n_ctx"],
        n_embd=ll_cfg["d_model"],
        n_layer=ll_cfg["n_layers"],
        n_head=ll_cfg["n_heads"],
        n_inner=ll_cfg["d_mlp"],  # TODO: check if this is correct
        activation_function=ll_cfg["act_fn"],
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        layer_norm_epsilon=ll_cfg["eps"],
        architectures=["GPT2LMHeadModel"],
        # not sure if we need to change these...
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.0,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
    )

    gpt = TLModel(model, config=my_config, output_type=output_type)
    gpt.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    type_to_module_mapping[type(gpt)] = tl_to_module_mapping
    type_to_dimension_mapping[type(gpt)] = tl_to_dimension_mapping

    return gpt, tokenizer


def make_hf_wrapper_from_tl_model(
    model: HookedTransformer, device="cpu", output_type=output_type.base_model_output
):
    model.to(device)
    ll_cfg = model.cfg.to_dict()
    my_config = GPT2Config(
        vocab_size=ll_cfg["d_vocab"],
        n_positions=ll_cfg["n_ctx"],
        n_embd=ll_cfg["d_model"],
        n_layer=ll_cfg["n_layers"],
        n_head=ll_cfg["n_heads"],
        n_inner=ll_cfg["d_mlp"],  # TODO: check if this is correct
        activation_function=ll_cfg["act_fn"],
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        layer_norm_epsilon=ll_cfg["eps"],
        architectures=["GPT2LMHeadModel"],
        # not sure if we need to change these...
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.0,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
    )

    hf_model = TLModel(model, config=my_config, output_type=output_type)
    hf_model.to(device)

    type_to_module_mapping[type(hf_model)] = tl_to_module_mapping
    type_to_dimension_mapping[type(hf_model)] = tl_to_dimension_mapping

    return hf_model
