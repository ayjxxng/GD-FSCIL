import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from PIL import Image
# ImageNet-A is the version defined at https://github.com/zhoudw-zdw/RevisitingCIL from here:
#   @article{zhou2023revisiting,
#        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
#        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
#        journal = {arXiv preprint arXiv:2303.07338},
#        year = {2023}
#    }

DATA_ROOT = './Data/DomainNet'


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args, isCifar=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        if isCifar:
            size = input_size
        else:
            size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(
                size, interpolation=Image.BICUBIC # TODO: transforms.InterpolationMode.BICUBIC -> BICBIC
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t


class domainnet(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(345).tolist()

    def __init__(self, inc, use_input_norm):
        self.inc = inc
        if use_input_norm:
            self.common_trsf = [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

    def download_data(self):
        # download from http://ai.bu.edu/M3SDA/#dataset (use "cleaned version")
        aa = np.loadtxt("./data/DomainNet/" + self.inc + "_train.txt", dtype="str")
        self.train_data = np.array(["./data/DomainNet/" + x for x in aa[:, 0]])
        self.train_targets = np.array([int(x) for x in aa[:, 1]])

        dil_tasks = ["real", "quickdraw", "painting", "sketch", "infograph", "clipart"]
        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt("./data/DomainNet/" + task + "_test.txt", dtype="str")
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.test_data = np.array(["./data/DomainNet/" + x for x in files])
        self.test_targets = np.array([int(x) for x in labels])


class DGFSCIL(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(345).tolist()

    def __init__(self, inc, first_source, use_iput_norm):
        self.inc = inc
        self.first_source = first_source
        self.fs_root = os.path.join("./data/DomainNet", f"fs_{self.first_source}")

        if use_iput_norm:
            self.common_trsf = [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

    def download_data(self):
        dil_tasks = ["real", "infograph", "painting", "sketch"]

        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt(os.path.join(self.fs_root, f"{task}_train.txt"), dtype="str")
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.train_data = np.array([f"{DATA_ROOT}/" + x for x in files])
        self.train_targets = np.array([int(x) for x in labels])

        dil_tasks = ["clipart", "quickdraw"]
        files = []
        labels = []
        for task in dil_tasks:
            aa = np.loadtxt(os.path.join(self.fs_root, f"{task}_test.txt"), dtype="str")
            files += list(aa[:, 0])
            labels += list(aa[:, 1])
        self.test_data = np.array([f"{DATA_ROOT}/" + x for x in files])
        self.test_targets = np.array([int(x) for x in labels])
