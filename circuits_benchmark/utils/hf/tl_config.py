from pyvene.models.constants import *


permute_tl_head_act = lambda x, *args, **kwargs: x.permute(0, 2, 1, 3)
combine_qkv = lambda x, *args, **kwargs: x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

tl_to_module_mapping = {
    "block_input": ("model.blocks[%s]", CONST_INPUT_HOOK),
    "block_output": ("model.blocks[%s]", CONST_OUTPUT_HOOK),
    # "mlp_activation": ("model.blocks[%s].mlp.act", CONST_OUTPUT_HOOK),
    "mlp_output": ("model.blocks[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("model.blocks[%s].mlp", CONST_INPUT_HOOK),

    # "attention_value_output": ("model.blocks[%s].attn.hook_v", CONST_INPUT_HOOK),
    # "head_attention_value_output": ("model.blocks[%s].attn.hook_v", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    # "attention_weight": ("model.blocks[%s].attn.attn_dropout", CONST_INPUT_HOOK),

    "attention_output": ("model.blocks[%s].attn.hook_result", CONST_OUTPUT_HOOK),
    "attention_input": ("model.blocks[%s].attn", CONST_INPUT_HOOK),

    # "query_output": ("model.blocks[%s].attn.hook_q", CONST_OUTPUT_HOOK, (combine_qkv, None)),
    # "key_output": ("model.blocks[%s].attn.hook_k", CONST_OUTPUT_HOOK, (combine_qkv, None)),
    # "value_output": ("model.blocks[%s].attn.hook_v", CONST_OUTPUT_HOOK, (combine_qkv, None)),
    
    # "head_query_output": ("model.blocks[%s].attn.hook_q", CONST_OUTPUT_HOOK, (permute_tl_head_act, None)), 
    # "head_key_output": ("model.blocks[%s].attn.hook_k", CONST_OUTPUT_HOOK, (permute_tl_head_act, None)),
    # "head_value_output": ("model.blocks[%s].attn.hook_v", CONST_OUTPUT_HOOK, (permute_tl_head_act, None)),
}

tl_to_dimension_mapping = {
    "n_head": ("n_head", ),
    "block_input": ("n_embd",),
    "block_output": ("n_embd",),
    "mlp_activation": (
        "n_inner",
        "n_embd*4",
    ),
    "mlp_output": ("n_embd",),
    "mlp_input": ("n_embd",),
    "attention_value_output": ("n_embd",),
    "head_attention_value_output": ("n_embd/n_head",),
    "attention_weight": ("max_position_embeddings", ),
    "attention_output": ("n_embd",),
    "attention_input": ("n_embd",),
    "query_output": ("n_embd",),
    "key_output": ("n_embd",),
    "value_output": ("n_embd",),
    "head_query_output": ("n_embd/n_head",),
    "head_key_output": ("n_embd/n_head",),
    "head_value_output": ("n_embd/n_head",),
}