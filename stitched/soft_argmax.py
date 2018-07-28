import torch

class SoftArgMax1D(torch.nn.Module):
    """
    Implementation of a soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....

        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).

        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:

        SoftArgMax(x) = \sum_i (i * softmax(x)_i)

        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size)
        return torch.matmul(smax, indices)