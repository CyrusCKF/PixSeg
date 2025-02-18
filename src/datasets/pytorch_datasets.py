import sys
from pathlib import Path

from torchvision import datasets

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_registry import register_dataset, register_metadata
from dataset_zoo import DatasetMeta

# fmt: off
VOC_LABELS = ("background",
              "person", 
              "bird", "cat", "cow", "dog", "horse", "sheep", 
              "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", 
              "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor")
VOC_COLORS = ((0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
              (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), 
              (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), 
              (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128))
CITYSCAPES_LABELS = ("road", "sidewalk", 
                     "building", "wall", "fence", 
                     "pole", "traffic light", "traffic sign", 
                     "vegetation", "terrain", 
                     "sky", 
                     "person", "rider", 
                     "car", "truck", "bus", "train", "motorcycle", "bicycle",
                     "background")
CITYSCAPES_COLORS = ((128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                     (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                     (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                     (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                     (0, 0, 230), (119, 11, 32), (0, 0, 0))
# fmt: on

register_metadata("VOC", DatasetMeta(21, 255, VOC_LABELS, VOC_COLORS))
register_metadata(
    "Cityscapes", DatasetMeta(20, 255, CITYSCAPES_LABELS, CITYSCAPES_COLORS)
)

# register builtin datasets
register_dataset(
    {"image_set": "train"},
    {"image_set": "val"},
    name="VOC",
)(datasets.VOCSegmentation)

register_dataset(
    {"mode": "segmentation", "image_set": "train"},
    {"mode": "segmentation", "image_set": "val"},
    name="SBD",
    meta_key="VOC",
)(datasets.SBDataset)

register_dataset(
    {"target_type": "semantic", "split": "train"},
    {"target_type": "semantic", "split": "val"},
)(datasets.Cityscapes)


def _test():
    from dataset_registry import DATASET_ZOO

    entry = DATASET_ZOO["VOC"]
    train_dataset = entry.construct_train(root=r"D:\_Dataset", year="2007")
    print(len(train_dataset))  # type: ignore
    print(entry.meta)


if __name__ == "__main__":
    _test()
