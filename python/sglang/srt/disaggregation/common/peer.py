"""Protocol-side record of one PD decode registration, held by prefill.

The protocol keys its per-peer state (rooms, blacklist, layout rejections) by
Peer and plans transfers from its PeerLayout; the Transport sees only `remote`.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.disaggregation.transport.base import RemoteBuffers

# Arbitrary; far above how long a failed write may still touch memory
# (about 5 s for Mooncake), so the release needs no in-flight tracking.
RETIRED_PEER_RELEASE_DELAY_S = 60.0
# Bounds how often the room table is scanned while a retired peer waits.
_RETIRED_PEER_SWEEP_INTERVAL_S = 1.0


class PeerLayout(msgspec.Struct, frozen=True):
    """What a decode reports for the protocol to plan transfers: topology and
    buffer geometry, never addresses. Lists are parallel to the decode's
    buffers of each kind; state lists are parallel to its state_types."""

    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_kv_layer_ids: List[int]
    dst_state_layer_ids: List[List[int]]
    dst_dcp_size: int = 1
    dst_dcp_rank: int = 0
    dst_draft_rows_dcp_collapsed: bool = False
    dst_kv_item_lens: List[int] = []
    dst_aux_item_lens: Optional[List[int]] = None
    # Zero when the decode has no staging ring.
    staging_total_size: int = 0
    # Staging slots stay [all K, all V] after draft buffers alter the KV order;
    # older peers leave this empty and callers fall back to dst_kv_layer_ids.
    staging_slot_layer_ids: List[int] = []
    # The receiver's registered draft-entry count. None when the peer predates
    # the field; a PP>1 sender stage without a draft runner cannot derive the
    # host-backed/device-only boundary from its own registration.
    num_draft_entries: Optional[int] = None


class Peer(msgspec.Struct, frozen=True, eq=False):
    """One decode registration as prefill holds it; compared by identity.

    A re-registration builds a new Peer, so state keyed by Peer never mixes two
    registrations of the same decode. A backend subclasses it to add what its
    send paths read from the registration.
    """

    # The name the decode's room messages carry; resolved to the Peer on arrival.
    name: str
    # The import result; the only part passed to the Transport.
    remote: RemoteBuffers
    # Layout-check result; only a re-registration lifts it.
    rejection: Optional[str]
    layout: PeerLayout
    # Per-entry token widths when the decode's DCP layout needs a relayout.
    dcp_token_item_lens: Optional[List[int]]

    @property
    def requires_dcp_relayout(self) -> bool:
        return self.dcp_token_item_lens is not None


class RetiredPeers:
    """Peers a re-registration replaced, waiting for their import to be released.

    Release is conservative: a retired peer is released once
    RETIRED_PEER_RELEASE_DELAY_S has passed since it retired and no room in the
    room table references it. Not thread-safe; the thread that registers peers
    owns it.
    """

    def __init__(self, release_remote: Callable[[RemoteBuffers], None]):
        self._release_remote = release_remote
        # (retire time, peer), oldest first.
        self._peers: List[Tuple[float, Peer]] = []
        self._next_sweep = 0.0

    def retire(self, peer: Peer) -> None:
        self._peers.append((time.monotonic(), peer))

    def sweep(self, transfer_infos: Dict[int, Dict[Peer, object]]) -> None:
        """Release the retired peers that are due. `transfer_infos` is the
        room table, keyed by room and then by Peer."""
        now = time.monotonic()
        if (
            not self._peers
            or now < self._next_sweep
            or now - self._peers[0][0] < RETIRED_PEER_RELEASE_DELAY_S
        ):
            return
        self._next_sweep = now + _RETIRED_PEER_SWEEP_INTERVAL_S
        # Snapshot: other threads add and drop rooms while this walks them.
        rooms = list(transfer_infos.values())
        waiting = []
        for retired_at, peer in self._peers:
            if now - retired_at < RETIRED_PEER_RELEASE_DELAY_S or any(
                peer in room for room in rooms
            ):
                waiting.append((retired_at, peer))
            else:
                self._release_remote(peer.remote)
        self._peers = waiting
