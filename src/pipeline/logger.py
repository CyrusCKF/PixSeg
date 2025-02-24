import logging
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import wandb
import wandb.wandb_run
from PIL.Image import Image
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.v2 import functional as TF

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.utils import visual

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
    """Base class for all loggers

    Contain hooks that can be overriden to save results during training
    """

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback) -> None:
        pass

    def on_running_metrics_updated(
        self, job_metrics: dict[str, dict[str, list[float]]]
    ):
        """
        Args:
            job_metrics: Mapping from each job to all metrics from the start of the training
        """
        pass

    def on_job_epoch_ended(
        self, job: str, step: int, cm: np.ndarray, metrics: dict[str, float]
    ):
        """
        Args:
            job: usually `train` or `val`
            cm: confusion matrix
        """
        pass

    def on_snapshots_created(self, job: str, step: int, snapshots: list[list[Tensor]]):
        """Snapshots are created after running each job epoch"""

    def on_checkpoint_saved(self, model_file: Path, checkpoint_file: Path):
        pass


class LocalLogger(Logger):
    CKPT_FOLDER = "ckpt"

    def __init__(self, folder: Path | None, labels: Sequence[str]) -> None:
        self.folder = folder
        self.labels = labels

    def __enter__(self):
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

    def on_running_metrics_updated(
        self, job_metrics: dict[str, dict[str, list[float]]]
    ):
        if self.folder is None:
            return
        visual.plot_running_metrics(job_metrics)
        visual.exhibit_figure(save_to=self.folder / "running_metrics.png")

    def on_job_epoch_ended(
        self, job: str, step: int, cm: np.ndarray, metrics: dict[str, float]
    ):
        if self.folder is None:
            return
        job_folder = self.folder / job
        job_folder.mkdir(exist_ok=True)

        normalized_cm = cm / cm.sum(axis=1, keepdims=True)
        visual.plot_confusion_matrix(normalized_cm, self.labels)
        visual.exhibit_figure(save_to=job_folder / f"cm_{step:>04}.png")

    def on_snapshots_created(self, job: str, step: int, snapshots: list[list[Tensor]]):
        if self.folder is None:
            return
        job_folder = self.folder / job
        job_folder.mkdir(exist_ok=True)
        path = self.folder / job / f"snapshot_{step:>04}.png"

        flat_snapshots = [s for ss in snapshots for s in ss]
        combined = visual.combine_images(flat_snapshots, nrow=3)
        combined_pil: Image = TF.to_pil_image(combined)
        combined_pil.save(path)


class WandbLogger(Logger):
    # TODO also save snapshots optionally
    def __init__(
        self,
        api_key: str | None,  # used for resuming
        run_id: str | None = None,
        **kwargs,
    ) -> None:
        """See :func:`wandb.init` for all supported kwargs

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

    def on_job_epoch_ended(
        self, job: str, step: int, cm: np.ndarray, metrics: dict[str, float]
    ):
        if self.run is None:
            return
        metrics_with_job = {job + "/" + k: v for k, v in metrics.items()}
        self.run.log(metrics_with_job, step=step)


class TensorboardLogger(Logger):
    # TODO use add_hparams to log config
    def __init__(
        self, parent_dir: str | None = None, save_images=True, **kwargs
    ) -> None:
        """See :class:`SummaryWriter` for all supported kwargs

        Args:
            parent_dir: dir to store all tensorboard logs. Sub folder is generated using
                the same scheme of :class:`SummaryWriter`
        """
        super().__init__()
        self.parent_dir = parent_dir
        if "log_dir" in kwargs and self.parent_dir is not None:
            logger.info("log_dir found in Tensorboard params, ignoring parent_dir")
            self.parent_dir = None
        self.save_images = save_images
        self.writer: SummaryWriter | None = None
        self.kwargs = kwargs

    def __enter__(self):
        if self.parent_dir is not None:
            # use the same default naming strategy in SummaryWriter
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            subfolder_name = f"{current_time}_{socket.gethostname()}"
            self.kwargs["log_dir"] = str(Path(self.parent_dir) / subfolder_name)
        self.writer = SummaryWriter(**self.kwargs)

    def __exit__(self, type, value, traceback) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def on_job_epoch_ended(
        self, job: str, step: int, cm: np.ndarray, metrics: dict[str, float]
    ):
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(job + "/" + k, v, step)

    def on_snapshots_created(self, job: str, step: int, snapshots: list[list[Tensor]]):
        if self.writer is None or not self.save_images:
            return
        for i, image_set in enumerate(snapshots):
            combined = visual.combine_images(image_set)
            self.writer.add_image(f"{job}_p{i}", combined, step)
