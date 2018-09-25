import torch
import torch.nn as nn
import torch.nn.functional as F





class TinyMultiTaskResNet(nn.Module):
    """
    A helper class. Implements a small resnet, with multiple outputs at the end.
    A fully connected encoder and decoder are specific instantiations of this class.
    Assumes a ReLU activation on all hidden units and identity activation on output.

    All hiden layers have a dimension of 'hidden_size'. The network has a stem of a single fully connected layer applied
    to the input. There are then 'num_blocks' many residual blocks, each with 'layers_per_block' many fully connected
    layers in them. We then allow multiple outputs, which are specified with dimensions in the list 'output_sizes'.

    Instantiations of this network can be used for a Fully Connected ResNet Encoder/Decoder/Generator/Discriminator.
    """
    def __init__(self, input_size=48, hidden_size=256, output_sizes=[10,10], layers_per_block=2, num_blocks=1):
        super(TinyMultiTaskResNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes
        self.layers_per_block = layers_per_block
        self.num_blocks = num_blocks

        self.input_fc = nn.Linear(input_size, hidden_size)

        blocks = []
        for _ in range(num_blocks):
            layers = []
            for _ in range(layers_per_block):
                fc = nn.Linear(hidden_size, hidden_size)
                bn = nn.BatchNorm1d(hidden_size)
                layers.append(nn.ModuleList([fc, bn]))
            blocks.append(nn.ModuleList(layers))
        self.blocks = nn.ModuleList(blocks)

        output_fcs = []
        for output_size in output_sizes:
            output_fcs.append(nn.Linear(hidden_size, output_size))
        self.output_fcs = nn.ModuleList(output_fcs)


    def forward(self, x):
        x = self.input_fc(x)

        for block in self.blocks:
            res = x
            for layer in block:
                fc, bn = layer[0], layer[1]
                x = fc(x)
                x = bn(x)
                x = F.relu(x)
            x = res + x

        outputs = []
        for output_fc in self.output_fcs:
            outputs.append(output_fc(x))

        return outputs