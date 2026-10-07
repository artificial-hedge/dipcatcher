"""SYNTHETIC adversarial probes for the uploads store — eviction
tombstones across journal replay, LRU recency (not FIFO), and the
declared-byte budget as a hard bound."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.serve.uploads import UPLOAD_MAX_PARTS, UploadMeta, UploadStore, UploadStoreError


def _store(state_dir: Path, max_entries: int = 2, max_bytes: int = 1 << 20) -> UploadStore:
    return UploadStore(max_entries=max_entries, max_bytes=max_bytes, state_dir=state_dir)


def _mk(store: UploadStore, name: str, nbytes: int = 4) -> UploadMeta:
    return store.create(
        purpose="fine-tune",
        filename=name,
        nbytes=nbytes,
        mime_type="application/octet-stream",
    )


def test_terminal_eviction_stays_evicted_across_restart(tmp_path: Path) -> None:
    """An evicted *completed* upload must carry a tombstone in the
    create payload — without ``evicted`` covering terminal records,
    replay restores them and the bounded store lies."""
    store = _store(tmp_path, max_entries=2)
    done = _mk(store, "done.bin")
    part = store.add_part(done.upload_id, b"ok!!")
    store.complete(done.upload_id, [part["id"]], content=b"ok!!", file_id="file-done")
    live = _mk(store, "live.bin")
    _mk(store, "churn.bin")  # pushes the oldest (done) over the bound

    assert store.get(done.upload_id) is None
    revived = _store(tmp_path, max_entries=2)
    assert revived.get(done.upload_id) is None  # tombstone held
    assert revived.get(live.upload_id) is not None


def test_pending_eviction_replays_as_expired(tmp_path: Path) -> None:
    """A pending upload squeezed out by churn must come back ``expired``
    (410), not silently resurrected and not fabricated as 404."""
    store = _store(tmp_path, max_entries=2)
    victim = _mk(store, "victim.bin")
    _mk(store, "b.bin")
    _mk(store, "c.bin")  # evicts victim (pending)

    revived = _store(tmp_path, max_entries=2)
    meta = revived.get(victim.upload_id)
    assert meta is not None and meta.status == "expired"
    with pytest.raises(UploadStoreError) as exc:
        revived.add_part(victim.upload_id, b"x")
    assert exc.value.code == "upload_expired"


def test_active_upload_outlives_dormant_on_eviction(tmp_path: Path) -> None:
    """Bounded LRU, not FIFO: a part arrival refreshes recency, so the
    dormant upload — never the active one — is the eviction victim."""
    store = _store(tmp_path, max_entries=2)
    active = _mk(store, "active.bin")
    dormant = _mk(store, "dormant.bin")
    store.add_part(active.upload_id, b"aa")  # refresh recency
    _mk(store, "churn.bin")

    assert store.get(active.upload_id) is not None
    assert store.get(dormant.upload_id) is None
    revived = _store(tmp_path, max_entries=2)
    assert revived.get(active.upload_id) is not None
    # dormant was pending when evicted — replay tombstones it as
    # ``expired`` (410), the honest end of a lapsed intent.
    meta = revived.get(dormant.upload_id)
    assert meta is not None and meta.status == "expired"


def test_declared_bytes_is_a_hard_budget(tmp_path: Path) -> None:
    """Cumulative part bytes can never exceed the declared total, and a
    declared total above the store cap refuses at create."""
    store = _store(tmp_path, max_entries=4)
    with pytest.raises(UploadStoreError) as exc:
        _mk(store, "too-big.bin", nbytes=(1 << 20) + 1)
    assert exc.value.code == "file_too_large"

    up = _mk(store, "parts.bin", nbytes=4)
    store.add_part(up.upload_id, b"abc")
    with pytest.raises(UploadStoreError) as exc2:
        store.add_part(up.upload_id, b"de")  # 3 + 2 > 4 declared
    assert exc2.value.code == "part_exceeds_declared_bytes"
    with pytest.raises(UploadStoreError) as exc3:
        store.add_part(up.upload_id, b"")
    assert exc3.value.code == "invalid_request"


def test_part_cap_and_missing_parts_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path, max_entries=4)
    up = _mk(store, "many.bin", nbytes=UPLOAD_MAX_PARTS + 1)
    for _ in range(UPLOAD_MAX_PARTS):
        store.add_part(up.upload_id, b"x")
    with pytest.raises(UploadStoreError) as exc:
        store.add_part(up.upload_id, b"y")
    assert exc.value.code == "too_many_parts"
    with pytest.raises(UploadStoreError) as exc2:
        store.assemble(up.upload_id, ["part_missing"])
    assert exc2.value.code == "part_not_found"


def test_complete_requires_exact_declared_bytes(tmp_path: Path) -> None:
    """A refused completion never mutates state — the upload stays
    pending; a wrong-byte assembly fails ``upload_incomplete``."""
    store = _store(tmp_path, max_entries=4)
    up = _mk(store, "exact.bin", nbytes=4)
    part = store.add_part(up.upload_id, b"abcd")
    with pytest.raises(UploadStoreError) as exc:
        store.complete(up.upload_id, [part["id"]], content=b"abc", file_id="file-x")
    assert exc.value.code == "upload_incomplete"
    assert store.get(up.upload_id).status == "pending"


def test_completed_upload_is_terminal(tmp_path: Path) -> None:
    """A completed upload can never re-enter pending — cancel 409s and
    extra parts 409, with the file_id binding intact across restart."""
    store = _store(tmp_path, max_entries=4)
    up = _mk(store, "sealed.bin", nbytes=2)
    part = store.add_part(up.upload_id, b"ok")
    store.complete(up.upload_id, [part["id"]], content=b"ok", file_id="file-sealed")

    with pytest.raises(UploadStoreError) as exc:
        store.cancel(up.upload_id)
    assert exc.value.code == "upload_terminal"
    with pytest.raises(UploadStoreError) as exc2:
        store.add_part(up.upload_id, b"!!")
    assert exc2.value.code == "upload_terminal"

    revived = _store(tmp_path, max_entries=4)
    meta = revived.get(up.upload_id)
    assert meta is not None
    assert meta.status == "completed"
    assert meta.file_id == "file-sealed"
    assert meta.parts == {}  # blobs dropped at completion
