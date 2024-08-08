import pickle
from typing import Dict, Set, Tuple, Optional, Literal

from iit.model_pairs.nodes import LLNode, HLNode
from iit.utils import index
from iit.utils.correspondence import Correspondence
from iit.utils.index import TorchIndex
from tracr.compiler.compiling import TracrOutput
from tracr.craft.bases import BasisDirection, VectorSpaceWithBasis
from tracr.craft.transformers import SeriesWithResiduals, MLP, MultiAttentionHead

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.iit.tracr_hl_node import TracrHLNode

TracrHLNodeMappingInfo = Tuple[int, Literal["attn", "mlp"], Optional[int | TorchIndex]]  # (layer, attn_or_mlp, head_index)

# TODO: We shouldn't define overrides using basis directions as keys, since sometimes Tracr uses multiple HL nodes for
#  the same basis.
tracr_corr_override_info: Dict[str, Dict[Tuple[str, Optional[str]], TracrHLNodeMappingInfo]] = {
    "3": {
            ("is_x_3", None): {(0, "mlp", index.Ix[[None]])},
            ("frac_prevs_1", None): {(1, "attn", index.Ix[:, :, 2, :])},
    }
}


class TracrCorrespondence(Correspondence):
    def __setattr__(self, __name: TracrHLNode, __value: set[LLNode]) -> None:
        if __name == "suffixes":
            assert isinstance(__value, dict), ValueError(
                f"__value is not a dict, but {type(__value)}"
            )
        else:
            assert isinstance(__name, TracrHLNode), ValueError(
                f"__name is not a TracrHLNode, but {type(__name)}"
            )
            assert isinstance(__value, set), ValueError(
                f"__value is not a set, but {type(__value)}"
            )
            assert all(isinstance(v, LLNode) for v in __value), ValueError(
                f"__value contains non-LLNode elements"
            )
        super().__setattr__(__name, __value)

    @classmethod
    def _build_corr_combining_info(
        cls,
        tracr_base_corr: Dict[BasisDirection, Set[TracrHLNodeMappingInfo]],
        tracr_corr_override: Optional[Dict[Tuple[str, Optional[str]], TracrHLNodeMappingInfo]] = None,
        hook_name_style: str = "tl",
    ):
        """This method builds a Tracr correspondence combining the basic information that Tracr provides (i.e. which
        basis directions are output by which components) and an optional mapping overriding the default Tracr info
         (e.g., to map specific basis directions to specific components).
        """
        def hook_name(loc, style) -> str:
            layer, attn_or_mlp, unit = loc
            assert attn_or_mlp in ["attn", "mlp"], ValueError(
                f"Unknown attn_or_mlp {attn_or_mlp}"
            )
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
            assert attn_or_mlp in ["attn", "mlp"], ValueError(
                f"Unknown attn_or_mlp {attn_or_mlp}"
            )
            if attn_or_mlp == "attn":
                return index.Ix[:, :, unit, :]
            assert unit is None
            return index.Ix[[None]]

        corr_dict: dict[HLNode, set[LLNode]] = {}
        for basis_dir, hl_locs in tracr_base_corr.items():
            assert tracr_corr_override is None or len(hl_locs) == 1, \
                "Tracr cases that have multiple HL nodes for the same basis direction are not supported when using overrides."

            for hl_loc in hl_locs:
                hl_node = TracrHLNode(
                            hook_name(hl_loc, hook_name_style),
                            label=basis_dir.name,
                            num_classes=0,  # TODO: get num_classes
                            idx=idx(hl_loc),
                        )

                ll_nodes = set()
                if tracr_corr_override is None:
                    # No override, just add the default mapping
                    ll_nodes.add(
                        LLNode(
                            hook_name(hl_loc, hook_name_style),
                            idx(hl_loc),
                            None,
                        )
                    )
                else:
                    for ll_loc in tracr_corr_override[basis_dir.name, basis_dir.value]:
                        ll_nodes.add(LLNode(hook_name(ll_loc, "tl"), idx(ll_loc)))

                corr_dict[hl_node] = ll_nodes

        return cls(corr_dict)

    @classmethod
    def make_identity_corr(cls, tracr_output: TracrOutput):
        """Creates a Tracr correspondence that maps each basis direction to a single (default) component."""
        return cls._build_corr_combining_info(cls.build_tracr_base_corr(tracr_output),
                                              None)

    @classmethod
    def from_output(cls, case: BenchmarkCase, tracr_output: TracrOutput):
        """Creates a Tracr correspondence from a Tracr output, using the given case to determine any overrides to the
        default Tracr correspondence info."""
        return cls._build_corr_combining_info(cls.build_tracr_base_corr(tracr_output),
                                              cls.get_tracr_corr_override_info(case))

    @classmethod
    def load(cls, filename: str):
        return cls(pickle.load(open(filename, "rb")))

    @classmethod
    def print_craft_model(cls, craft_model: SeriesWithResiduals):
        def names(vsb: VectorSpaceWithBasis):
            return [(bd.name, bd.value) for bd in vsb.basis]

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


    @classmethod
    def build_tracr_base_corr(
        cls,
        tracr_output: TracrOutput
    ) -> Dict[BasisDirection, Set[TracrHLNodeMappingInfo]]:
        """Builds the basic Tracr correspondence information from the Tracr output."""
        craft_model: SeriesWithResiduals = tracr_output.craft_model
        result: Dict[BasisDirection, Set[TracrHLNodeMappingInfo]] = {}

        i = 0
        for block in craft_model.blocks:
            if isinstance(block, MLP):
                assert block.fst.output_space == block.snd.input_space
                for direction in block.snd.output_space.basis:
                    if direction not in result:
                        result[direction] = set()

                    result[direction].add((i, "mlp", None))

                i += 1
            elif isinstance(block, MultiAttentionHead):
                for j, sb in enumerate(block.sub_blocks):
                    for direction in sb.w_ov.output_space.basis:
                        if direction not in result:
                            result[direction] = set()

                        result[direction].add((i, "attn", j))

            else:
                raise ValueError(f"Unknown block type {type(block)}")

        return result

    @classmethod
    def get_tracr_corr_override_info(
        cls,
        case: BenchmarkCase
    ) -> Dict[Tuple[str, Optional[str]], TracrHLNodeMappingInfo] | None:
        """Returns the Tracr correspondence override info for the given case, if any."""
        if case.get_name() in tracr_corr_override_info.keys():
            return tracr_corr_override_info[case.get_name()]
        return None
