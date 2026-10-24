"""Execute launchers using isolated test doubles, never real training/simulation."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherExecutionTests(unittest.TestCase):
    def test_training_check_only_and_batch_rejection(self):
        with tempfile.TemporaryDirectory(prefix="train-launch-") as tmp:
            root = Path(tmp)
            resources = root / "resources with spaces"
            resources.mkdir()
            manifest = root / "manifest.txt"
            manifest.write_text("fixture /unused\n")
            env = dict(os.environ, TRAIN_ENV=str(Path(sys.executable).parent.parent),
                       OUT_DIR=str(root / "output"), LOG_DIR=str(root / "logs"),
                       BASE_HF_CKPT=str(resources), TOKENIZER_PATH=str(resources),
                       DATA_MANIFEST=str(manifest), ROBOT_CONFIG_ROOT=str(resources),
                       MOTION_WEIGHTS=str(resources), GPU_IDS="0,1,2,3,4,5,6,7",
                       CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", CHECK_ONLY="1",
                       BASE_YAML=str(ROOT / "configs/vla/robotwin/robotwin.yaml"))
            for key in ("MOGE_PATH", "MORGBD_PATH", "DINO_VIDEO_CKPT", "DINO_VIDEO_CFG"):
                env[key] = str(manifest)
            command = ["bash", str(ROOT / "scripts/train_stage2.sh")]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("batch=64", result.stdout)
            self.assertFalse((root / "output").exists())
            for overrides, expected in (({"GLOBAL_BS": "512"}, "global batch"),
                                        ({"NPROC_PER_NODE": "1"}, "NPROC_PER_NODE"),
                                        ({"GPU_IDS": "0,1", "CUDA_VISIBLE_DEVICES": "0,1"}, "8 卡")):
                result = subprocess.run(command, cwd=root, env=dict(env, **overrides), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout + result.stderr)

    def run_eval_fixture(self, episodes, fail=False, no_video=False):
        with tempfile.TemporaryDirectory(prefix="eval-launch-") as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            sim = root / "sim directory"
            (sim / "script").mkdir(parents=True)
            model = root / "model with ' quote"
            model.mkdir()
            conda = root / "conda profile.sh"
            conda.write_text('conda() { return 0; }\nexport PATH=' + shlex.quote(str(bin_dir)) + ':"$PATH"\n')
            fake = bin_dir / "python"
            fake.write_text('#!' + sys.executable + '\n' + '''import json, os, sys, time
from pathlib import Path
args=sys.argv[1:]
if args[:1] == ['--version']:
    print('fixture interpreter'); sys.exit(0)
if args[:1] == ['-c']:
    print(os.environ['FIXTURE_SITE']); sys.exit(0)
kind='server' if args[:1] == ['-m'] else 'client'
Path(os.environ['FIXTURE_ROOT'], kind+'.json').write_text(json.dumps({'argv':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))
if kind == 'server':
    time.sleep(60)
else:
    print('Success rate: 1/'+os.environ['FIXTURE_EPISODES']+' => 1.0%')
    sys.exit(int(os.environ['FIXTURE_FAIL']))
''')
            fake.chmod(0o755)
            env = dict(os.environ, MODEL_PATH=str(model), EVAL_WORKDIR=str(sim), OUTPUT_BASE=str(root / "results"),
                       CONDA_SH=str(conda), GPU_IDS="2,5", CUDA_VISIBLE_DEVICES="2,5",
                       FIXTURE_ROOT=str(root), FIXTURE_SITE=str(root / "site"), FIXTURE_EPISODES=str(episodes),
                       FIXTURE_FAIL=str(int(fail)), DVAC_ARGS="--dvac_alpha 1.0 --dvac_tail_steps 3")
            cmd = ["bash", str(ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"),
                   "--num_tasks", "1", "--instruction_type", "unseen"]
            if no_video:
                cmd.append("--no_video")
            result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 3 if fail or episodes != 100 else 0, result.stdout + result.stderr)
            server = json.loads((root / "server.json").read_text())
            client = json.loads((root / "client.json").read_text())
            self.assertEqual(server['gpu'], '2')
            self.assertEqual(client['gpu'], '2')
            self.assertEqual(server['argv'][server['argv'].index('--model_path')+1], str(model))
            self.assertEqual(server['argv'][server['argv'].index('--dvac_alpha')+1], '1.0')
            self.assertEqual(client['argv'][client['argv'].index('--eval_video_log')+1], 'False' if no_video else 'True')
            self.assertEqual(client['argv'][client['argv'].index('--instruction_type')+1], 'unseen')

    def test_eval_full_fixture_and_argument_forwarding(self):
        self.run_eval_fixture(100)

    def test_eval_partial_success_exit_is_rejected(self):
        self.run_eval_fixture(4, no_video=True)

    def test_eval_exhausted_retries_are_rejected(self):
        self.run_eval_fixture(100, fail=True)


if __name__ == '__main__':
    unittest.main()
