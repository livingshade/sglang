"""PD control-plane version negotiation.

Each prefill rank advertises on ``PUT /route`` its transfer backend and two
inclusive ``(min, max)`` ranges: ``wire_versions`` for the shared header layout
(the header's ``Ver``) and ``transport_versions`` for its transport's payloads
(the header's ``Trans Version``). Decode reads them from ``GET /route`` and picks
the highest common version of each before sending any ZMQ message. Version 0 of
both is each backend's legacy positional frame format.
"""

from __future__ import annotations

from typing import Optional, Tuple

VersionRange = Tuple[int, int]

# Header layouts every transport speaks.
WIRE_VERSIONS: VersionRange = (0, 0)

# A prefill that predates negotiation omits both ranges and speaks only legacy.
PRE_NEGOTIATION_VERSIONS: VersionRange = (0, 0)


def to_version_range(versions) -> VersionRange:
    lowest, highest = versions
    return int(lowest), int(highest)


def negotiate_version(
    *, name: str, local: VersionRange, remote: Optional[VersionRange]
) -> int:
    """The highest version both ranges contain."""
    if remote is None:
        remote = PRE_NEGOTIATION_VERSIONS
    lowest = max(local[0], remote[0])
    highest = min(local[1], remote[1])
    if lowest > highest:
        raise RuntimeError(
            f"No common PD {name} version: prefill speaks {list(remote)}, decode "
            f"speaks {list(local)}. Upgrade the older server so both ranges overlap."
        )
    return highest
