# Unlearn-only pipeline: no relearning attack. Returns best few-shot robustness metric.
# Usage: python3 src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepCollapse task_name=test
import os
import signal
import shutil
import subprocess
import uuid
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def _get_run_name(cfg: DictConfig) -> str:
    try:
        job_num = HydraConfig.get().job.num
        return f"{cfg.task_name}_{job_num}"
    except Exception:
        return cfg.task_name


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg.trainer.args.run_name = _get_run_name(cfg)

    suffix = cfg.task_name + "_" + uuid.uuid4().hex[:8]
    cfg.paths.tmp_comm_dir = str(Path(cfg.paths.tmp_comm_dir) / suffix)
    comm_dir = Path(cfg.paths.tmp_comm_dir)
    comm_dir.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: exit(1))

    try:
        if "UNL_WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = os.environ["UNL_WANDB_PROJECT"]
        unlearning_cfg_path = comm_dir / "unlearning_cfg.yaml"
        OmegaConf.save(cfg, unlearning_cfg_path)
        subprocess.run(
            [
                "python3",
                "src/train.py",
                f"--config-path={comm_dir.absolute()}",
                "--config-name=unlearning_cfg.yaml",
            ],
            check=True,
        )

        # Return the optimisation metric from the last valid eval
        robustness = float(open(comm_dir / "robustness.txt").read())
        print(f"Robustness: {robustness}")
        return robustness

    finally:
        shutil.rmtree(comm_dir)


if __name__ == "__main__":
    main()
