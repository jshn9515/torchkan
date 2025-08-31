import torch
import torch.nn as nn
from torch import Tensor


class MatryoshkaHead(nn.Module):
    def __init__(
        self,
        nesting_list: list[int],
        num_classes: int = 1000,
        efficient: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes  # Number of classes for classification
        self.efficient = efficient
        self.classifier: list[nn.Linear] = []
        if self.efficient:
            self.classifier.append(
                nn.Linear(nesting_list[-1], self.num_classes, **kwargs),
            )
        else:
            for num_features in self.nesting_list:
                self.classifier.append(
                    nn.Linear(num_features, self.num_classes, **kwargs),
                )

    def reset_parameters(self):
        if self.efficient:
            self.classifier[0].reset_parameters()
        else:
            for net in self.classifier:
                net.reset_parameters()

    def forward(self, x: Tensor) -> list[Tensor]:
        nesting_logits = []
        for i, num_features in enumerate(self.nesting_list):
            if self.efficient:
                if self.classifier[0].bias is None:
                    nesting_logits.append(
                        torch.matmul(
                            x[:, :num_features],
                            (self.classifier[0].weight[:, :num_features]).T,
                        ),
                    )
                else:
                    nesting_logits.append(
                        torch.matmul(
                            x[:, :num_features],
                            (self.classifier[0].weight[:, :num_features]).T,
                        )
                        + self.classifier[0].bias,
                    )
            else:
                nesting_logits.append(self.classifier[i](x[:, :num_features]))

        return nesting_logits
