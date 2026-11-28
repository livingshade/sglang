"""PD control-plane version negotiation.

Each prefill rank advertises on ``PUT /route`` its transfer backend and, as two
parallel lists, the header layout versions (the header's ``Ver``) it speaks and
the inclusive ``[min, max]`` transport payload versions (``Trans Version``) it
speaks under each. Decode reads them from ``GET /route`` and picks the highest
common ``Ver`` whose transport ranges overlap, then the highest common
``Trans Version`` within it, before sending any ZMQ message. ``Ver`` 0 is each
backend's legacy headerless frame format, and only ``Trans Version`` 0 goes with it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

VersionRange = Tuple[int, int]
# Header Ver -> inclusive (min, max) Trans Version spoken under it.
SupportedVersions = Dict[int, VersionRange]

# Also what a prefill that predates negotiation speaks.
LEGACY_VERSIONS: SupportedVersions = {0: (0, 0)}


def encode_route_versions(
    versions: SupportedVersions,
) -> Tuple[List[int], List[List[int]]]:
    """The ``(wire_versions, transport_versions)`` lists sent on ``/route``."""
    wire_versions = sorted(versions)
    return wire_versions, [list(versions[ver]) for ver in wire_versions]


def decode_route_versions(
    wire_versions: Optional[List[int]], transport_versions: Optional[List[List[int]]]
) -> Optional[SupportedVersions]:
    """None when the prefill predates negotiation and sent neither list."""
    if wire_versions is None:
        return None
    return {
        int(ver): (int(lowest), int(highest))
        for ver, (lowest, highest) in zip(
            wire_versions, transport_versions, strict=True
        )
    }


def negotiate_versions(
    *, local: SupportedVersions, remote: Optional[SupportedVersions]
) -> Tuple[int, int]:
    """The ``(Ver, Trans Version)`` pair to speak with the remote."""
    if remote is None:
        remote = LEGACY_VERSIONS
    for ver in sorted(local.keys() & remote.keys(), reverse=True):
        lowest = max(local[ver][0], remote[ver][0])
        highest = min(local[ver][1], remote[ver][1])
        if lowest <= highest:
            return ver, highest
    raise RuntimeError(
        f"No common PD wire version: prefill speaks {remote}, decode speaks "
        f"{local} (header Ver -> Trans Version range). Upgrade the older server."
    )
