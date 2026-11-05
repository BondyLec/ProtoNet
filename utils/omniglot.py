# omniglot.py
# 合并Omniglot数据集并重新编号类别

from torch.utils.data import Dataset
from torchvision import transforms, datasets

class OmniglotCombined(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x)
        ])

        # 训练集类别
        self.background = datasets.Omniglot(
            root=root,
            background=True,
            download=True,
            transform=self.transform
        )
        # 验证集类别
        self.evaluation = datasets.Omniglot(
            root=root,
            background=False,
            download=True,
            transform=self.transform
        )

        self.background_num_classes = len(
            self.background._characters
        )
        self.class_names = (
            list(self.background._characters) + list(self.evaluation._characters)
        )
        self.name_to_label = {
            name: label
            for label,name in enumerate(self.class_names)
        }
        self.total_length = (
            len(self.background)
            + len(self.evaluation)
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.background):
            x, y = self.background[idx]
        else:
            new_idx = idx - len(self.background)
            x, y =self.evaluation[new_idx]
            y = y + self.background_num_classes
        return x, y