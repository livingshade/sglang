"""PD control-plane wire version negotiation.

Each prefill rank advertises on ``PUT /route`` its transfer backend and the
inclusive ``(min, max)`` range of control-plane wire versions that backend
speaks. Decode reads both from ``GET /route`` and picks the highest common
version before sending any ZMQ message. Version 0 is each backend's legacy
positional frame format.
"""

from __future__ import annotations

from typing import Optional, Tuple

WireVersions = Tuple[int, int]

# A prefill that predates negotiation omits wire_versions and speaks only legacy.
PRE_NEGOTIATION_WIRE_VERSIONS: WireVersions = (0, 0)


def negotiate_wire_version(
    *, local: WireVersions, remote: Optional[WireVersions]
) -> int:
    """The highest wire version both ranges contain."""
    if remote is None:
        remote = PRE_NEGOTIATION_WIRE_VERSIONS
    lowest = max(local[0], remote[0])
    highest = min(local[1], remote[1])
    if lowest > highest:
        raise RuntimeError(
            f"No common PD wire version: prefill speaks {list(remote)}, decode "
            f"speaks {list(local)}. Upgrade the older server so both ranges overlap."
        )
    return highest
