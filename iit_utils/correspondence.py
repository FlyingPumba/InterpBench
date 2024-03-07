import networkx as nx
from tracr.compiler.compiling import TracrOutput
from tracr.craft.transformers import MLP, MultiAttentionHead, SeriesWithResiduals
from tracr.craft import bases
import iit.model_pairs as mp
from iit.utils import index


def names(vsb: bases.VectorSpaceWithBasis):
    return [(bd.name, bd.value) for bd in vsb.basis]


def print_craft_model(craft_model: SeriesWithResiduals):
    for block in craft_model.blocks:
        if isinstance(block, MLP):
            print("MLP:")
            print(names(block.fst.input_space), " -> ", names(block.fst.output_space))
            assert block.fst.output_space == block.snd.input_space
            print("\t -> ", names(block.snd.output_space))

        elif isinstance(block, MultiAttentionHead):
            print("MultiAttentionHead:")
            for i, sb in enumerate(block.sub_blocks):
                print(f"head {i}:")
                print(f"qk left {names(sb.w_qk.left_space)}")
                print(f"qk right {names(sb.w_qk.right_space)}")
                w_ov = sb.w_ov
                print(f"w_ov {names(w_ov.input_space)} -> \n\t{names(w_ov.output_space)}")

        else:
            raise ValueError(f"Unknown block type {type(block)}")
        print()


class TracrCorrespondence:
    """
    Stores a dictionary that takes tracr graph nodes to HookPoint nodes...
    """

    def __init__(self, graph: nx.DiGraph, craft_model: SeriesWithResiduals):
        self.graph = graph
        self.craft_model = craft_model

        self._dict = dict()
        i = 0
        for block in craft_model.blocks:
            if isinstance(block, MLP):
                assert block.fst.output_space == block.snd.input_space
                for direction in block.snd.output_space.basis:
                    self._dict[direction] = (i, "mlp", None)
                i += 1
            elif isinstance(block, MultiAttentionHead):
                for j, sb in enumerate(block.sub_blocks):
                    for direction in sb.w_ov.output_space.basis:
                        self._dict[direction] = (i, "attn", j)
                
            else:
                raise ValueError(f"Unknown block type {type(block)}")

        """
    This is not necessary for now, because node names and direction names are equal
    """
        # for name, node in graph.nodes.items():
        #   if name in ["indices", "tokens"]: continue
        #   if "OUTPUT_BASIS" in node:
        #     for direction in node["OUTPUT_BASIS"]:
        #       self._dict[direction] = self._unit_output_bases[direction]

    @staticmethod
    def from_output(tracr_output: TracrOutput):
        return TracrCorrespondence(tracr_output.graph, tracr_output.craft_model)

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self) -> str:
        dict_repr = "\n".join([f"\t{k.name, k.value}: {v}" for k, v in self._dict.items()])
        return f"TracrCorrespondence(\n{dict_repr}\n)"

    def items(self):
        return self._dict.items()


def make_hl_ll_corr(tracr_hl_corr, tracr_ll_corr, hook_name_style="tl") -> dict[mp.HLNode, set[mp.LLNode]]:
    def hook_name(loc, style) -> str:
        layer, attn_or_mlp, unit = loc
        assert attn_or_mlp in ["attn", "mlp"], ValueError(f"Unknown attn_or_mlp {attn_or_mlp}")
        if style == "tl":
            return f"blocks.{layer}.{attn_or_mlp}.{'hook_result' if attn_or_mlp == 'attn' else 'hook_post'}"
        elif style == "wrapper":
            return f"mod.blocks.{loc}.mod.{attn_or_mlp}.hook_point"
        else:
            raise ValueError(f"Unknown style {style}")

    def idx(loc):
        _, attn_or_mlp, unit = loc
        if isinstance(unit, index.TorchIndex):
            return unit
        assert attn_or_mlp in ["attn", "mlp"], ValueError(f"Unknown attn_or_mlp {attn_or_mlp}")
        if attn_or_mlp == "attn":
            return index.Ix[:, :, unit, :]
        assert unit is None
        return index.Ix[[None]]

    if tracr_ll_corr is None:
        print("WARNING: tracr_ll_corr is None, returning an Identity correspondence using HL Model")
        return {
            mp.HLNode(hook_name(hl_loc, hook_name_style), None, idx(hl_loc)): {
                mp.LLNode(hook_name(hl_loc, hook_name_style), idx(hl_loc), None)
            }
            for basis_dir, hl_loc in tracr_hl_corr.items()
        }

    return {
        mp.HLNode(hook_name(hl_loc, hook_name_style), None, idx(hl_loc)): {
            mp.LLNode(hook_name(ll_loc, "tl"), idx(ll_loc)) for ll_loc in tracr_ll_corr[basis_dir.name, basis_dir.value]
        }
        for basis_dir, hl_loc in tracr_hl_corr.items()
    }
