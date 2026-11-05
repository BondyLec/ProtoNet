import torch
import torch.nn as nn
from models.encoder import OmniglotEncoder

class Protonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = OmniglotEncoder()

    def forward(
            self,
            support_x,
            support_y,
            query_x,
            n_way
    ):
        # Support与Query合并经过同一Encoder
        # BatchNorm使用当前batch的μ与σ，Support与Query合并可避免两者标准化采用各自参数
        all_x = torch.cat(
            [support_x, query_x],
            dim=0
        )
        all_z = self.encoder(all_x)
        # 拆开Support与Query
        num_spport = support_x.size(0)
        support_z = all_z[:num_spport]
        query_z = all_z[num_spport:]
        # Support -> Prototype
        prototypes = []
        for class_id in range(n_way):
            class_embeddings = support_z[
                support_y == class_id
            ]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        # 平方欧式距离与logit计算
        distances = ((query_z.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=2)
        logits = -distances
        return logits