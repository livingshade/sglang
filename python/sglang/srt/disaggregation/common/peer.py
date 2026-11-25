"""Protocol-side record of one PD decode registration, held by prefill.

The protocol keys its per-peer state (rooms, blacklist, layout rejections) by
Peer; the Transport sees only `remote`.
"""

from __future__ import annotations

from typing import Optional

import msgspec

from sglang.srt.disaggregation.transport.base import RemoteBuffers


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
