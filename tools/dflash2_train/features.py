"""Target feature sources for DFlash2 training.

``FeatureServer`` drives one ``tqfeatures`` process (azllm target + drafter layer list loaded once)
over its line/frame protocol. ``FeaturePool`` fans requests out to several servers and yields
samples in completion order. ``FeatureCache`` stores frames on disk so a run can be replayed
without the target loaded (e.g. to train on the same card the server would otherwise occupy).
"""

from __future__ import annotations

import hashlib
import os
import queue
import struct
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

MAGIC = 0x31465154  # "TQF1"


@dataclass
class Sample:
    input_ids: np.ndarray  # int64 [P]
    loss_mask: np.ndarray  # float32 [P]
    hidden: np.ndarray  # uint16 (bf16 bits) [P, nf * H]


def _read_exact(f, n: int) -> bytearray:
    """Read a frame without reallocating chunks or copying the complete payload."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        count = f.readinto(view[pos:])
        if not count:
            raise EOFError("tqfeatures closed its output")
        pos += count
    return buf


class FeatureServer:
    def __init__(self, cmd: list[str], env: dict | None = None, log_path: str | None = None):
        self.log = open(log_path, "ab") if log_path else None
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log or None,
            env={**os.environ, **(env or {})},
            bufsize=0,
        )

    def send(self, ids: Iterable[int]) -> None:
        """Submit a request; replies must be read in the same order."""
        line = " ".join(str(int(t)) for t in ids) + "\n"
        self.proc.stdin.write(line.encode())

    def receive(self) -> np.ndarray | None:
        """Read the oldest response: [P, nf * H] uint16 BF16 bits, or None."""
        magic, P, nf, H = struct.unpack("<4I", _read_exact(self.proc.stdout, 16))
        if magic != MAGIC:
            raise RuntimeError(f"tqfeatures: bad frame magic {magic:#x}")
        if P == 0:
            return None
        raw = _read_exact(self.proc.stdout, P * nf * H * 2)
        return np.frombuffer(raw, dtype=np.uint16).reshape(P, nf * H)

    def features(self, ids: Iterable[int]) -> np.ndarray | None:
        self.send(ids)
        return self.receive()

    def close(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except BrokenPipeError:
                    pass
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            if self.proc.stdout:
                self.proc.stdout.close()
            if self.log:
                self.log.close()


class FeatureCache:
    """Frames keyed by a hash of the token ids, one .npy per sequence."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, ids: np.ndarray) -> str:
        h = hashlib.sha1(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()
        return os.path.join(self.root, h[:2], h + ".npy")

    def get(self, ids: np.ndarray) -> np.ndarray | None:
        p = self._path(ids)
        return np.load(p, mmap_mode="r") if os.path.exists(p) else None

    def put(self, ids: np.ndarray, hidden: np.ndarray) -> None:
        p = self._path(ids)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f".tmp{os.getpid()}.npy"
        np.save(tmp, hidden)
        os.replace(tmp, p)


class FeaturePool:
    """Feed (input_ids, loss_mask) records through N servers; yields Samples as they complete.

    With ``servers == []`` every record must already be in ``cache`` (replay mode).
    """

    def __init__(self, servers: list[FeatureServer], cache: FeatureCache | None = None, prefetch: int = 4):
        self.servers = servers
        self.cache = cache
        self.prefetch = max(1, prefetch)

    def run(self, records: Iterator[tuple[np.ndarray, np.ndarray]]) -> Iterator[Sample]:
        if not self.servers:
            for ids, mask in records:
                hid = self.cache.get(ids) if self.cache else None
                if hid is None:
                    raise KeyError("feature cache miss in replay mode (no servers)")
                yield Sample(ids, mask, hid)
            return

        todo: queue.Queue = queue.Queue(maxsize=self.prefetch * len(self.servers))
        done: queue.Queue = queue.Queue(maxsize=self.prefetch * len(self.servers))
        stop = object()
        errors: list[BaseException] = []

        def feeder():
            try:
                for rec in records:
                    todo.put(rec)
            finally:
                for _ in self.servers:
                    todo.put(stop)

        def worker(srv: FeatureServer):
            # Two outstanding requests let the target compute the next sequence while its
            # writer thread streams the previous frame. More outstanding requests can
            # deadlock if both stdin and stdout pipe buffers fill simultaneously.
            pending = deque()
            finished = False
            try:
                while not finished or pending:
                    while not finished and len(pending) < 2:
                        try:
                            rec = todo.get(block=not pending)
                        except queue.Empty:
                            break
                        if rec is stop:
                            finished = True
                            break
                        ids, mask = rec
                        hid = self.cache.get(ids) if self.cache else None
                        if hid is not None:
                            done.put(Sample(ids, mask, hid))
                            continue
                        srv.send(ids)
                        pending.append((ids, mask))
                    if not pending:
                        continue
                    ids, mask = pending.popleft()
                    hid = srv.receive()
                    if hid is not None and self.cache:
                        self.cache.put(ids, hid)
                    if hid is not None:
                        done.put(Sample(ids, mask, hid))
            except BaseException as e:  # surface in the consumer thread
                errors.append(e)
            finally:
                done.put(stop)

        threads = [threading.Thread(target=feeder, daemon=True)]
        threads += [threading.Thread(target=worker, args=(s,), daemon=True) for s in self.servers]
        for t in threads:
            t.start()
        live = len(self.servers)
        while live:
            item = done.get()
            if item is stop:
                live -= 1
                continue
            yield item
        if errors:
            raise errors[0]
