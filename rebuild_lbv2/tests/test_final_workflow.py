"""Small CPU fixtures exercise motion weights and real DCP/HF export."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("motion", ROOT / "tools/build_motion_weights.py")
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


class MotionTests(unittest.TestCase):
    def test_smoothing_never_crosses_episode_boundary(self):
        actions = np.concatenate([np.zeros((10, 1)), np.arange(10)[:, None]])
        weights, _ = motion.motion_weights(actions, [10, 10], window=4)
        np.testing.assert_allclose(weights[:10], np.repeat(weights[0], 10))
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_rejects_duplicate_frame_indices(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(prefix="motion-") as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            pq.write_table(pa.table({"episode_index": [0]*4, "frame_index": [0, 1, 1, 3],
                                     "action": [[0.0]]*4}), data / "episode.parquet")
            with self.assertRaises(ValueError):
                motion._read_actions(tmp, [4])

    def test_rejects_invalid_motion_inputs(self):
        for kwargs in ({"window": 0}, {"lo": -1}, {"hi": 0.1}, {"power": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                motion.motion_weights(np.ones((4, 1)), [4], **kwargs)
        with self.assertRaises(ValueError):
            motion.motion_weights(np.ones((4, 1)), [3])


class ExportTests(unittest.TestCase):
    def test_real_dcp_to_plain_and_ema_safetensors(self):
        with tempfile.TemporaryDirectory(prefix="export-") as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            (assets / "config.json").write_text('{"model_type":"fixture"}')
            (assets / "tokenizer_config.json").write_text('{}')
            # A stale index must never be copied alongside a new single shard.
            (assets / "model.safetensors.index.json").write_text('{"weight_map":{"old":"missing.safetensors"}}')
            for branch, value in (("model", 2.0), ("ema", 3.0)):
                expected = {"weight": torch.full((2, 3), value)}
                dcp.save({"model": expected}, checkpoint_id=root / "checkpoint" / branch)
                out = root / (branch + "_hf")
                command = [sys.executable, str(ROOT / "tools/export_hf_from_dcp.py"),
                           "--dcp", str(root / "checkpoint"), "--assets-from", str(assets), "--out", str(out)]
                if branch == "ema":
                    command.append("--ema")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                tensors = {}
                for shard in out.glob("*.safetensors"):
                    tensors.update(load_file(str(shard)))
                self.assertEqual(set(tensors), set(expected))
                torch.testing.assert_close(tensors["weight"], expected["weight"])
                self.assertEqual(json.loads((out / "config.json").read_text())["model_type"], "fixture")
                self.assertFalse((out / "model.safetensors.index.json").exists())


if __name__ == "__main__":
    unittest.main()
