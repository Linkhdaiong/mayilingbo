from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "scripts" / "train_stage2.sh").read_text(encoding="utf-8")
EVAL = (ROOT / "experiment" / "robotwin" / "start_robotwin_infer_and_eval.sh").read_text(encoding="utf-8")
PATHS = ROOT / "configs" / "local_paths.sh"


class FinalLauncherContractTests(unittest.TestCase):
    def test_training_uses_visible_gpu_count_and_global_batch_formula(self):
        self.assertIn("CUDA_VISIBLE_DEVICES", TRAIN)
        self.assertIn("NPROC_PER_NODE", TRAIN)
        self.assertIn('nproc-per-node="$NPROC_PER_NODE"', TRAIN)
        self.assertIn("MICRO_BS * GA * NPROC_PER_NODE", TRAIN)
        self.assertNotIn("--nproc-per-node=1", TRAIN)

    def test_training_sources_single_local_path_file(self):
        self.assertIn("local_paths.sh", TRAIN)
        self.assertTrue(PATHS.is_file())
        for name in ("BASE_HF_CKPT", "DATA_MANIFEST", "TOKENIZER_PATH", "OUT_DIR"):
            self.assertRegex(PATHS.read_text(encoding="utf-8"), re.compile(rf"^{name}=", re.MULTILINE))

    def test_eval_forwards_dvac_args_to_policy_server(self):
        self.assertIn("DVAC_ARGS", EVAL)
        self.assertIn("--chunk_ret", EVAL)
        self.assertIn('"${DVAC_ARGS_ARRAY[@]}"', EVAL)

    def test_eval_preserves_video_default_and_supports_gpu_ids(self):
        self.assertNotIn("keep_inference=false\nenable_video=False", EVAL)
        self.assertIn("GPU_IDS", EVAL)
        self.assertIn("gpu_ids[", EVAL)

    def test_eval_marks_exhausted_retries_incomplete(self):
        self.assertIn("incomplete=", EVAL)
        self.assertIn("incomplete=$((incomplete + 1))", EVAL)
        self.assertIn("exit 3", EVAL)
        self.assertNotIn("skipped=$((skipped + 1))", EVAL)


if __name__ == "__main__":
    unittest.main()
