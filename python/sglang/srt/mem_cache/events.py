# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV cache placement event recording.

Produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` events
consumed by KV-aware routers (e.g. dynamo). A cache holds one recorder and calls
it; the recorder owns the queue and needs nothing back from its owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)

if TYPE_CHECKING:
    from sglang.srt.lora.lora_registry import LoRARef


def _namespaced_chain_continues(*, tail: BlockStored, event: BlockStored) -> bool:
    """Whether ``event`` extends ``tail``'s namespaced chain, when one is emitted."""
    if tail.namespaced_block_hashes is None or event.namespaced_block_hashes is None:
        return (
            tail.namespaced_block_hashes is None
            and event.namespaced_block_hashes is None
        )
    return (
        bool(tail.namespaced_block_hashes)
        and event.namespaced_parent_block_hash == tail.namespaced_block_hashes[-1]
    )


class LoRANameTable:
    """Maps the lora_id at the end of a radix key's extra_key to its adapter name.

    Req appends lora_id to extra_key, so a tree node knows only the id. Entries
    are never dropped: ids are not reused, and nodes cached under an unloaded
    adapter can still publish stores until they are evicted.
    """

    def __init__(self):
        self._names: dict[str, str] = {}
        self._id_lengths: set[int] = set()

    @classmethod
    def from_lora_refs(cls, lora_refs: Optional[Iterable[LoRARef]]) -> LoRANameTable:
        table = cls()
        for lora_ref in lora_refs or ():
            table.register(lora_id=lora_ref.lora_id, lora_name=lora_ref.lora_name)
        return table

    def register(self, *, lora_id: str, lora_name: str) -> None:
        self._names[lora_id] = lora_name
        self._id_lengths.add(len(lora_id))

    def resolve(self, extra_key: Optional[str]) -> Optional[str]:
        if not extra_key:
            return None
        for id_length in self._id_lengths:
            lora_name = self._names.get(extra_key[-id_length:])
            if lora_name is not None:
                return lora_name
        return None


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        page_size: int,
        emit_namespaced_hashes: bool = False,
        lora_names: Optional[LoRANameTable] = None,
    ):
        self.enabled = enabled
        self.page_size = page_size
        self.emit_namespaced_hashes = emit_namespaced_hashes
        self.lora_names = lora_names if lora_names is not None else LoRANameTable()
        self._queue: list = []

    def enqueue(self, event) -> None:
        """Append an event, coalescing it with a compatible queue tail.

        KV event batches already support multiple block hashes.  Combining them
        here avoids emitting one event per page while preserving ordering and
        the parent-linked store chains consumers use to rebuild the cache tree.
        """
        if self._queue:
            tail = self._queue[-1]

            if isinstance(tail, BlockRemoved) and isinstance(event, BlockRemoved):
                if tail.medium == event.medium and (
                    (tail.namespaced_block_hashes is None)
                    == (event.namespaced_block_hashes is None)
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    if tail.namespaced_block_hashes is not None:
                        tail.namespaced_block_hashes.extend(
                            event.namespaced_block_hashes
                        )
                    return

            elif isinstance(tail, BlockStored) and isinstance(event, BlockStored):
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.lora_name == event.lora_name
                    and tail.block_size == event.block_size
                    and tail.cache_salt == event.cache_salt
                    and tail.session_id == event.session_id
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                    and _namespaced_chain_continues(tail=tail, event=event)
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    tail.token_ids.extend(event.token_ids)
                    if tail.namespaced_block_hashes is not None:
                        tail.namespaced_block_hashes.extend(
                            event.namespaced_block_hashes
                        )
                    return

        self._queue.append(event)

    def _node_event_hash_values(self, node: Any) -> list:
        """Hash values to publish for ``node``, computing them if not yet set."""
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        if node.key.extra_key is None and node.key.cache_salt is None:
            return node.hash_value
        return compute_node_event_hash_values(node, self.page_size)

    def _parent_block_hash(self, node: Any) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if node.key.extra_key is not None or node.key.cache_salt is not None:
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return hash_str_to_int64(parent_hash_values[-1])

    @staticmethod
    def _namespaced_parent_block_hash(node: Any) -> Optional[int]:
        """The storage-chain link of ``node``'s first page.

        Mirrors the parent condition of ``compute_node_hash_values``, so the
        link is exactly the hash the node's own storage chain continues from.
        """
        parent = node.parent
        if parent is None or not parent.hash_value or len(parent.key) == 0:
            return None
        return hash_str_to_int64(parent.hash_value[-1])

    def record_store(
        self, node: Any, medium=None, *, session_id: Optional[str] = None
    ) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        event_hash_values = self._node_event_hash_values(node)
        parent_block_hash = self._parent_block_hash(node)
        lora_name = self.lora_names.resolve(node.key.extra_key)
        namespaced_parent_block_hash = (
            self._namespaced_parent_block_hash(node)
            if self.emit_namespaced_hashes
            else None
        )

        page_index = 0
        logical_len = len(node.key)
        is_bigram = node.key.is_bigram
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            # Preserve historical event payload: bigram pages expose tuples.
            if is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = hash_str_to_int64(event_hash_values[page_index])
            namespaced_block_hash = (
                hash_str_to_int64(node.hash_value[page_index])
                if self.emit_namespaced_hashes
                else None
            )

            self.enqueue(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                    medium=medium,
                    cache_salt=node.key.cache_salt,
                    session_id=session_id,
                    lora_name=lora_name,
                    namespaced_block_hashes=(
                        [namespaced_block_hash] if self.emit_namespaced_hashes else None
                    ),
                    namespaced_parent_block_hash=namespaced_parent_block_hash,
                )
            )

            parent_block_hash = block_hash
            namespaced_parent_block_hash = namespaced_block_hash
            page_index += 1

    def record_remove(self, node: Any, medium=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        # Hash values must match what was stored.
        event_hash_values = self._node_event_hash_values(node)

        block_hashes = []
        namespaced_block_hashes = [] if self.emit_namespaced_hashes else None
        logical_len = len(node.key)
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(hash_str_to_int64(event_hash_values[page_index]))
            if namespaced_block_hashes is not None:
                namespaced_block_hashes.append(
                    hash_str_to_int64(node.hash_value[page_index])
                )
            page_index += 1

        if block_hashes:
            self.enqueue(
                BlockRemoved(
                    block_hashes=block_hashes,
                    medium=medium,
                    namespaced_block_hashes=namespaced_block_hashes,
                )
            )

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self.enqueue(AllBlocksCleared())

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enabled:
            return []
        events = self._queue
        self._queue = []
        return events
