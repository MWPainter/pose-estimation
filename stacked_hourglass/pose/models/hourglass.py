'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['HourglassNet']



def opt_pack_history(x, hist, using_attn, time):
    """
    Make packing and unpacking history a little nicer, so that we don't have to change the input to the
    nn.Module's everywhere.

    This "packs" the image input + history input together, to what the network is expecting.

    If there is no attention, it is expecting just the image input, if there is history, then
    """
    if using_attn:
        return (x, hist[time])
    else:
        return x



def opt_unpack_history(x_hist, using_attn):
    """
    Make packing and unpacking history a little nicer, so that we don't have to change the input to the
    nn.Module's everywhere.
    """
    if using_attn:
        return x_hist # = (x, hist)
    else:
        return (x_hist, None)





class _Attention(nn.Module):
    """
    PyTorch nn.Module that implements a general 1D attention layer.
    """
    def __init__(self, history_length, embedding_size):
        super(_Attention, self).__init__()
        self.history_length = history_length
        self.embedding_size = embedding_size
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y, history = x
        y = y.view((-1, self.embedding_size))                                       # size = (bs, es)
        history = history.view((-1, self.history_length, self.embedding_size))      # size = (bs, hl, es)
        weights = self.softmax(torch.matmul(self.fc(history), y))                   # size = (bs, hl, 1)
        weights = weights.expand(history.size())                                    # size = (bs, hl, es)
        attention_vector = torch.sum(weights * history, dim=1)                      # size = (bs, es)
        return attention_vector.view(y.size())




class _ResBlock(nn.Module):
    """
    3 Conv Res Block
    """
    expansion = 2
    def __init__(self, inchannels, channels, stride=1, downsample=None, use_layer_norm=False, width=0, height=0,
                 batch_norm_momentum=0.1, use_batch_norm_affine=True):
        """
        Initialize a residual block

        :param inchannels: Number of channels being input to the convolutions
        :param channels: Number of channels to be output by the convllutions (so this is the number of input channels for
            layers after the first)
        :param stride: Stride to be used in the convolutional modules
        :param downsample: An nn.Module used to downsample the image
        :param use_layer_norm: If we should replace the batch norm layers by layer norm layers (TODO: will break if stride != 1)
        :param width: The (spatial) width of the input, so that we can use layernorm
        :param height: The (spatial) height of the input, so that we can use layernorm
        :param batch_norm_momentum: Momentum term to use in batch norm layers
        :param use_batch_norm_affine: If we should use affine parameters in batch norm layers
        """
        super(_ResBlock, self).__init__()
        self.batch_norm_momentum = batch_norm_momentum
        self.use_batch_norm_affine = use_batch_norm_affine
        self.use_layer_norm = use_layer_norm
        self.size = [inchannels, width, height]
        self.size2 = [channels, width, height]

        if use_layer_norm:
            self.ln1 = nn.LayerNorm([inchannels, width, height], elementwise_affine=False)
            self.ln2 = nn.LayerNorm([channels, width, height], elementwise_affine=False)
            self.ln3 = nn.LayerNorm([channels, width, height], elementwise_affine=False)
        else:
            self.bn1 = nn.BatchNorm2d(inchannels, momentum=batch_norm_momentum, affine=self.use_batch_norm_affine)
            self.bn2 = nn.BatchNorm2d(channels, momentum=batch_norm_momentum, affine=self.use_batch_norm_affine)
            self.bn3 = nn.BatchNorm2d(channels, momentum=batch_norm_momentum, affine=self.use_batch_norm_affine)

        self.norm1 = self.ln1 if use_layer_norm else self.bn1
        self.conv1 = nn.Conv2d(inchannels, channels, kernel_size=1, bias=True)
        self.norm2 = self.ln2 if use_layer_norm else self.bn2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.norm3 = self.ln3 if use_layer_norm else self.bn3
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out





class _Hourglass(nn.Module):
    """
    Implements a single hourglass module. Applies 2 res blocks, recurses, and then 1 res block after upsampling the
    ourput of the recursion. At the lowest resolution we apply 3 res blocks, without any recursion.
    """
    def __init__(self, num_blocks, channels, depth, use_attention=False, attn_history_length=0, use_layer_norm=False,
                 width=0, height=0, batch_norm_momentum=0.1, use_batch_norm_affine=True):
        """
        :param block: nn.Module for a residual block to use
        :param num_blocks: Number of the above blocks to use in sequence
        :param channels: Number of channels in the convolution modules
        :param depth: How many times to reduce the spatial dimensions/recurse
        :param use_attention: if we are using attentiojn for sequential data (at the lowest resolution/the 'bottleneck')
        :param attn_history_length: length of the history to use for attention
        :param use_layer_norm: If we should replace the batch norm layers by layer norm layers
        :param width: The (spatial) width of the input, so that we can use layernorm
        :param height: The (spatial) height of the input, so that we can use layernorm
        :param batch_norm_momentum: Momentum term to use in batch norm layers
        :param use_batch_norm_affine: If we should use affine parameters in batch norm layers
        """
        super(_Hourglass, self).__init__()
        self.batch_norm_momentum = batch_norm_momentum
        self.use_batch_norm_affine = use_batch_norm_affine
        self.use_layer_norm = use_layer_norm
        self.width = width
        self.height = height
        self.use_attention = use_attention
        self.attn_history_length = attn_history_length

        self.depth = depth
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(num_blocks, channels, depth, width, height)

        if use_attention:
            self.attention = _Attention(attn_history_length, 100)# TODO: use the correct size rather than '100'
            self.attention_reduction = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=True)



    def _make_residual(self, num_blocks, channels, width=0, height=0):
        """
        Stacks multiple residual blocks
        """
        layers = []
        for i in range(0, num_blocks):
            layers.append(_ResBlock(channels*_ResBlock.expansion, channels, use_layer_norm=self.use_layer_norm, width=width,
                                    height=height, batch_norm_momentum=self.batch_norm_momentum,
                                    use_batch_norm_affine=self.use_batch_norm_affine))
        return nn.Sequential(*layers)



    def _make_hour_glass(self, num_blocks, channels, depth, width, height):
        """
        Construct the hourglass module

        hg will be used from depth-1, depth-2, down to 1
        at each depth, the spatial dims are halved
        so at index i, the input dimensions will be width // 2^(depth-1+i) and height // 2^(depth-1+i)

        depth-1 is only used on the initial input, so we only make one residual block for that
        (because the hourglass network has explicit residual blocks between the hourglasses, that are applied at
        this resolution).

        :param num_blocks: Number of resblocks to stack
        :param channels: The number of channels to use in each conv layer
        :param depth: The depth to recurse to
        :return: Returns a module list of nn.Sequentials. It contains all of the resblocks to use (see _hour_glass_forward)
        """
        hg = []
        for i in range(depth-1):
            res = []
            for j in range(3):
                res.append(self._make_residual(num_blocks, channels, width//(2**(depth-1-i)), height//(2**(depth-1-i))))
            hg.append(nn.ModuleList(res))
        hg.append(self._make_residual(num_blocks, channels, width, height))
        return nn.ModuleList(hg)



    def _hour_glass_forward(self, n, x, hist):
        """
        The hourglass forward computation (recursive)
        """
        if n == self.depth:
            # At the highest resolution, only apply one conv here
            x = self.hg[n-1](x)

            # Applying hourglass at lower resolutions (recursive)
            if n > 1:
                y = F.max_pool2d(x, 2, stride=2)
                y = self._hour_glass_forward(n-1, y, hist)
                y = self.upsample(y)

            return x + y

        else:
            # Apply two res blocks at this resolution
            x = self.hg[n-1][0](x)
            x = self.hg[n-1][1](x)

            # Apply hourglass at lower resolutions (recursive)
            if n > 1:
                y = F.max_pool2d(x, 2, stride=2)
                y = self._hour_glass_forward(n-1, y, hist)
                y = self.upsample(y)
                x = x + y

            # TODO: Add in the attention stuff ("if n == 1:")

            # One last res block at this resolution
            x = self.hg[n-1][2](x)
            return x



    def forward(self, x):
        """
        Forward pass of the hourglass module.
        Decouples logic to encode
        """
        x, hist = opt_unpack_history(x, self.use_attention)
        return self._hour_glass_forward(self.depth, x, hist)





class HourglassNet(nn.Module):
    """
    Hourglass model from Newell et al ECCV 2016
    """
    def __init__(self, num_stacks=2, num_blocks=4, num_classes=16, use_attention=False, attn_history_length=0,
                 use_layer_norm=False, width=0, height=0, batch_norm_momentum=0.1, no_batch_norm_affine=False):
        super(HourglassNet, self).__init__()

        self.batch_norm_momentum = batch_norm_momentum
        self.use_batch_norm_affine = not no_batch_norm_affine
        self.use_layer_norm = use_layer_norm
        self.width = width
        self.height = height
        self.use_attention = use_attention
        self.attn_history_length = attn_history_length

        self.inchannels = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.num_classes = num_classes

        # Build the 'stem' of the network
        self.conv1 = nn.Conv2d(3, self.inchannels, kernel_size=7, stride=2, padding=3,
                               bias=True)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm([self.inchannels, width//2, height//2], elementwise_affine=False)
        else:
            self.bn1 = nn.BatchNorm2d(self.inchannels, momentum=batch_norm_momentum, affine=self.use_batch_norm_affine)
        self.norm1 = self.ln1 if use_layer_norm else self.bn1

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(self.inchannels, 1, width=width//2, height=height//2)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer2 = self._make_residual(self.inchannels, 1, width=width//4, height=height//4)
        self.layer3 = self._make_residual(self.num_feats, 1, width=width//4, height=height//4)

        # build hourglass modules (see 'forward' for how they all work and come together)
        ch = self.num_feats*_ResBlock.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(_Hourglass(num_blocks, self.num_feats, 4, use_attention=use_attention,
                                 use_layer_norm=use_layer_norm, width=width//4, height=height//4,
                                 batch_norm_momentum=batch_norm_momentum, use_batch_norm_affine=self.use_batch_norm_affine))

            res.append(self._make_residual(self.num_feats, num_blocks, width=width//4, height=height//4))
            fc.append(self._make_fc(ch, ch, width//4, height//4))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))

            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)



    def _make_residual(self, channels, blocks, width=0, height=0, stride=1):
        """
        Make a residual block, used between hourglass modules at the highest resolutions.
        """
        downsample = None
        if stride != 1 or self.inchannels != channels * _ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannels, channels * _ResBlock.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(_ResBlock(self.inchannels, channels, stride, downsample, use_layer_norm=self.use_layer_norm,
                                width=width, height=height, batch_norm_momentum=self.batch_norm_momentum))
        self.inchannels = channels * _ResBlock.expansion
        for i in range(1, blocks):
            layers.append(_ResBlock(self.inchannels, channels, use_layer_norm=self.use_layer_norm, width=width, height=height,
                                    batch_norm_momentum=self.batch_norm_momentum))

        return nn.Sequential(*layers)



    def _make_fc(self, inchannels, outchannels, width=0, height=0):
        """
        A 1x1 conv layer. (With some normalization and relu activation).
        """
        # Make the normalization nn.Module. (Note the constructors side effects, adding to the hourglasses parameter list)
        if self.use_layer_norm:
            ln = nn.LayerNorm([outchannels, width, height], elementwise_affine=False)
        else:
            bn = nn.BatchNorm2d(inchannels, momentum=self.batch_norm_momentum, affine=self.use_batch_norm_affine)
        norm = ln if self.use_layer_norm else bn

        # Return the 1x1 conv
        conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                norm,
                self.relu,
            )



    def scale_weights_(self, factor):
        """
        Scale all of the weights in this network by 'factor'.
        """
        for p in self.parameters():
            if p.requires_grad:
                p.data *= factor



    def forward(self, x):
        """
        Forward pass
        :param x: (stacked_hourglass, history) input to the network, and (if using attention) the history of the low
            dimension representations (from the last T inputs). history[i] is the last T 'embeddings' from hourglass i.
        :return: Result of the forward pass, a list (of length = number of stacks) of score outputs
        """
        out = []
        x, hist = opt_unpack_history(x, self.use_attention)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            y = opt_pack_history(x, hist, self.use_attention, i)
            y = self.hg[i](y)                                           # hourglass module
            y = self.res[i](y)                                          # an additional residual block
            y = self.fc[i](y)                                           # a "fc block", which is just a 1x1 reduction conv
            score = self.score[i](y)                                    # predict scores at this layer
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)                                    # get features from the "fc block" (match the current dims of x)
                score_ = self.score_[i](score)                          # get features from this score (match the current dims of x)
                x = x + fc_ + score_                                    # input for the next hourglass, including a residual connection over the whole hourglass

        return out


