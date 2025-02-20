import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import PIL.Image
import wandb.wandb_run

import wandb

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.utils import visual
from src.utils.metrics import MetricStore

logger = logging.getLogger(__name__)


@contextmanager
def init_logging(log_file: Path | None):
    """Init logging with clearer format. Output to console and file if set

    Closes all handlers when done
    """
    FORMAT = r"%(asctime)s :: %(name)s.%(levelname)-8s :: %(message)s"
    DATEFMT = r"%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers[0].setLevel(logging.INFO)
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        handlers[-1].setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG, format=FORMAT, datefmt=DATEFMT, handlers=handlers
    )

    # these loggers are too annoying. Hide them
    loggers = [
        "PIL.TiffImagePlugin",
        "PIL.PngImagePlugin",
        "PIL.Image",
        "matplotlib.colorbar",
        "matplotlib.pyplot",
        "matplotlib.font_manager",
    ]
    for l in loggers:
        logging.getLogger(l).propagate = False

    try:
        yield
    finally:
        logging.root.handlers.clear()


class Logger:
    """Base class for all loggers"""

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback) -> None:
        pass

    def log_running_metrics(self, metrics: dict[str, dict[str, list[float]]]):
        """Called whenever there is an update in metrics of each job"""
        pass

    def log_epoch_metrics(self, job: str, step: int, ms: MetricStore):
        """Called every epoch for each job"""
        pass

    def save_snapshot(self, job: str, step: int, snapshot: PIL.Image.Image):
        """Called every epoch for each job"""
        pass


class LocalLogger(Logger):
    CKPT_FOLDER = "ckpt"

    def __init__(self, folder: Path | None, labels: Sequence[str]) -> None:
        self.folder = folder
        self.labels = labels

    def __enter__(self):
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

    def log_running_metrics(self, metrics: dict[str, dict[str, list[float]]]):
        if self.folder is None:
            return
        visual.plot_running_metrics(metrics)
        visual.exhibit_figure(save_to=self.folder / "running_metrics.png")

    def log_epoch_metrics(self, job: str, step: int, ms: MetricStore):
        if self.folder is None:
            return
        job_folder = self.folder / job
        job_folder.mkdir(exist_ok=True)

        cm = ms.confusion_matrix
        normalized_cm = cm / cm.sum(axis=1, keepdims=True)
        visual.plot_confusion_matrix(normalized_cm, self.labels)
        visual.exhibit_figure(save_to=job_folder / f"cm_{step:>04}.png")

    def save_snapshot(self, job: str, step: int, snapshot: PIL.Image.Image):
        if self.folder is None:
            return
        job_folder = self.folder / job
        job_folder.mkdir(exist_ok=True)
        path = self.folder / job / f"snapshot_{step:>04}.png"
        snapshot.save(path)


class WandbLogger(Logger):
    def __init__(
        self,
        api_key: str | None,  # used for resuming
        run_id: str | None = None,
        **kwargs,
    ) -> None:
        """See :func:`wandb.init` for all supported parameters

        Set :param:`run_id` to resume wandb logging
        """
        os.environ["WANDB_SILENT"] = "true"  # set it before init to avoid logging that
        self.api_key = api_key
        self.run_id = run_id
        self.kwargs = kwargs
        self.run: wandb.wandb_run.Run | None = None

    def __enter__(self):
        mode = "disabled" if self.api_key is None else "online"
        if self.api_key is not None:
            wandb.login(key=self.api_key, verify=True)

        # disable unnecessary stuff
        settings = wandb.Settings(
            console="off", disable_git=True, x_save_requirements=False
        )
        self.run = wandb.init(
            mode=mode,
            id=self.run_id,
            resume="allow",
            settings=settings,
            **self.kwargs,
        )
        logger.info(f"Wandb run id: {self.run.id}")

    def __exit__(self, type, value, traceback) -> None:
        wandb.finish()

    def log_epoch_metrics(self, job: str, step: int, ms: MetricStore):
        if self.run is None:
            return
        metrics = ms.summarize()
        metrics_with_job = {job + "/" + k: v for k, v in metrics.items()}
        self.run.log(metrics_with_job, step=step)


def _test():
    with init_logging(Path("run.log")):
        logging.info("Hi")


if __name__ == "__main__":
    _test()
