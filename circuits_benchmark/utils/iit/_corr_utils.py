import networkx as nx

from tracr.compiler.compiling import TracrOutput
from tracr.craft import bases
from tracr.craft.transformers import MLP, MultiAttentionHead, SeriesWithResiduals


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


class TracrHLCorr: # TODO: remove this class
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
        return TracrHLCorr(tracr_output.graph, tracr_output.craft_model)

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self) -> str:
        dict_repr = "\n".join([f"\t{k.name, k.value}: {v}" for k, v in self._dict.items()])
        return f"TracrCorrespondence(\n{dict_repr}\n)"

    def items(self):
        return self._dict.items()

