"""Framed feature transport: fragmented reads, EOF, pipeline ordering and rejection."""
import io
import pathlib
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "dflash2_train"))
from features import MAGIC, FeaturePool, FeatureServer, _read_exact  # noqa: E402


class PartialReader(io.BytesIO):
    def readinto(self, b):
        return super().readinto(b[: min(3, len(b))])


FAKE_SERVER = r"""
import struct, sys
for line in sys.stdin:
    ids = [int(t) for t in line.split()]
    if ids[0] == 999:
        sys.stdout.buffer.write(struct.pack('<4I', 0x31465154, 0, 0, 0))
    else:
        values = [ids[0], len(ids)] * len(ids)
        sys.stdout.buffer.write(struct.pack('<4I', 0x31465154, len(ids), 1, 2))
        sys.stdout.buffer.write(struct.pack('<%dH' % len(values), *values))
    sys.stdout.buffer.flush()
"""


class TransportTests(unittest.TestCase):
    def test_readinto_handles_short_reads_and_avoids_full_copy(self):
        data = bytes(range(17))
        raw = _read_exact(PartialReader(data), len(data))
        self.assertIsInstance(raw, bytearray)
        self.assertEqual(raw, data)

    def test_read_exact_eof(self):
        with self.assertRaises(EOFError):
            _read_exact(PartialReader(b"abc"), 6)

    def test_feature_server_and_pipeline_order(self):
        server = FeatureServer([sys.executable, "-u", "-c", FAKE_SERVER])
        try:
            tokens = [np.array([i] * (i + 2), dtype=np.int64) for i in range(1, 7)]
            masks = [np.ones(len(ids), dtype=np.float32) for ids in tokens]
            pool = FeaturePool([server], prefetch=4)
            got = list(pool.run(iter(zip(tokens, masks))))
            self.assertEqual(len(got), len(tokens))
            for sample, ids in zip(got, tokens):
                self.assertTrue(np.array_equal(sample.input_ids, ids))
                self.assertEqual(sample.hidden.shape, (len(ids), 2))
                self.assertTrue(np.all(sample.hidden[:, 0] == ids[0]))
                self.assertTrue(np.all(sample.hidden[:, 1] == len(ids)))
            self.assertIsNone(server.features([999]))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
