# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
# And taken from https://github.com/davidmrau/mixture-of-experts with modifications

import torch
from torch import Tensor


class SparseDispatcher:
    def __init__(self, num_experts: int, gates: Tensor):
        self.gates = gates
        self.num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(dim=0)
        # drop indices
        _, self.expert_index = torch.split(sorted_experts, 1, dim=1)
        # get according batch index for each expert
        self.batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self.part_sizes = torch.sum(gates > 0, dim=0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self.batch_index.flatten()]
        self.nonzero_gates = torch.gather(gates_exp, dim=1, index=self.expert_index)

    def dispatch(self, inp: Tensor) -> tuple[Tensor, ...]:
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self.batch_index].squeeze(1)
        return torch.split(inp_exp, self.part_sizes, dim=0)

    def combine(
        self, expert_out: list[Tensor], conv_dims: int, multiply_by_gates: bool = True
    ):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.concat(expert_out, dim=0)
        tuple_conv_dims = (1,) * conv_dims
        if multiply_by_gates:
            nonzero_gates = self.nonzero_gates.view(
                self.nonzero_gates.shape + tuple_conv_dims
            )
            stitched = stitched * nonzero_gates
        out_size = (self.gates.size(0),) + expert_out[-1].shape[1:]
        zeros = torch.zeros(out_size, requires_grad=True, device=expert_out[-1].device)

        # combine samples that have been processed by the same k experts
        combined = torch.index_add(zeros, 0, self.batch_index, stitched.float())
        return combined

    def expert_to_gates(self) -> tuple[Tensor, ...]:
        # split nonzero gates for each expert
        return torch.split(self.nonzero_gates, self.part_sizes, dim=0)
