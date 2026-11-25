"""What a Mooncake prefill holds about each decode registration."""

from __future__ import annotations

import struct
from typing import List, Optional

import msgspec

from sglang.srt.disaggregation.common.peer import Peer, PeerLayout
from sglang.srt.disaggregation.common.utils import unpack_int_lists
from sglang.srt.disaggregation.mooncake.buffers import BufferKeys
from sglang.srt.disaggregation.mooncake.transport import MooncakeRemoteBuffers


class KVArgsRegisterInfo(msgspec.Struct, frozen=True):
    """A decode's registration message: what the transport imports, and the
    PeerLayout the protocol plans transfers from."""

    mooncake_session_id: str
    dst_kv_ptrs: List[int]
    dst_aux_ptrs: List[int]
    dst_state_data_ptrs: List[List[int]]  # parallel to state_types
    staging_base_ptr: int
    layout: PeerLayout

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> KVArgsRegisterInfo:
        layout = PeerLayout(
            dst_aux_item_lens=(
                list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])) if msg[6] else None
            ),
            dst_tp_rank=int(msg[8].decode("ascii")),
            dst_attn_tp_size=int(msg[9].decode("ascii")),
            dst_kv_item_len=int(msg[10].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[12], "I") if len(msg) > 12 else []
            ),
            dst_kv_layer_ids=(
                list(struct.unpack(f"{len(msg[13]) // 4}I", msg[13]))
                if len(msg) > 13 and msg[13] != b""
                else []
            ),
            dst_state_layer_ids=(
                unpack_int_lists(msg[14], "I")
                if len(msg) > 14 and msg[14] != b""
                else []
            ),
            staging_total_size=(
                int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else 0
            ),
            dst_dcp_size=(
                int(msg[17].decode("ascii")) if len(msg) > 17 and msg[17] != b"" else 1
            ),
            dst_dcp_rank=(
                int(msg[18].decode("ascii")) if len(msg) > 18 and msg[18] != b"" else 0
            ),
            staging_slot_layer_ids=(
                list(struct.unpack(f"{len(msg[19]) // 8}Q", msg[19]))
                if len(msg) > 19 and msg[19]
                else []
            ),
            dst_kv_item_lens=(
                list(struct.unpack(f"{len(msg[20]) // 8}Q", msg[20]))
                if len(msg) > 20 and msg[20]
                else []
            ),
            dst_draft_rows_dcp_collapsed=(len(msg) > 21 and msg[21] == b"1"),
            num_draft_entries=(
                int(msg[22].decode("ascii"))
                if len(msg) > 22 and msg[22] != b""
                else None
            ),
        )
        return cls(
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[7], "Q"),
            staging_base_ptr=(
                struct.unpack("Q", msg[15])[0]
                if len(msg) > 15 and len(msg[15]) == 8
                else 0
            ),
            layout=layout,
        )


class MooncakePeer(Peer, frozen=True, eq=False):
    """A Mooncake decode registration; `name` is its mooncake session id."""

    remote: MooncakeRemoteBuffers
    keys: BufferKeys
    # Per-entry token widths when the decode's DCP layout needs a relayout.
    dcp_token_item_lens: Optional[List[int]]
    # Registration cleared the state strides; skip the per-transfer check.
    state_strides_validated: bool

    @property
    def requires_dcp_relayout(self) -> bool:
        return self.dcp_token_item_lens is not None
