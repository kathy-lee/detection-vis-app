import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from typing import Optional
    

class RecordEncoder(nn.Module):
    def __init__(self, in_channels, config, norm='layer'):
        """
        RECurrent Online object detectOR (RECORD) features extractor.
        @param in_channels: number of input channels (default: 8)
        @param config: number of input channels per block
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordEncoder, self).__init__()
        self.norm = norm
        # Set the number of input channels in the configuration file
        config['in_conv']['in_channels'] = in_channels

        # config_tmp = **config['in_conv'] --> dict with arguments

        # Input convolution (expands the number of input channels)
        self.in_conv = Conv3x3ReLUNorm(in_channels=config['in_conv']['in_channels'],
                                       out_channels=config['in_conv']['out_channels'],
                                       stride=config['in_conv']['stride'], norm=norm)

        # IR block 1 (acts as a bottleneck)
        self.ir_block1 = self._make_ir_block(in_channels=config['ir_block1']['in_channels'],
                                             out_channels=config['ir_block1']['out_channels'],
                                             num_block=config['ir_block1']['num_block'],
                                             expansion_factor=config['ir_block1']['expansion_factor'],
                                             stride=config['ir_block1']['stride'], use_norm=config['ir_block1']['use_norm'])

        # IR block 2 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block2 = self._make_ir_block(in_channels=config['ir_block2']['in_channels'],
                                             out_channels=config['ir_block2']['out_channels'],
                                             num_block=config['ir_block2']['num_block'],
                                             expansion_factor=config['ir_block2']['expansion_factor'],
                                             stride=config['ir_block2']['stride'], use_norm=config['ir_block2']['use_norm'])

        # Bottleneck LSTM 1 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm1']['use_norm'] else self.norm
        self.bottleneck_lstm1 = BottleneckLSTM(input_channels=config['bottleneck_lstm1']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm1']['out_channels'],
                                               norm=lstm_norm)

        # IR block 3 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block3 = self._make_ir_block(in_channels=config['ir_block3']['in_channels'],
                                             out_channels=config['ir_block3']['out_channels'],
                                             num_block=config['ir_block3']['num_block'],
                                             expansion_factor=config['ir_block3']['expansion_factor'],
                                             stride=config['ir_block3']['stride'], use_norm=config['ir_block3']['use_norm'])

        # Bottleneck LSTM 2 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm2']['use_norm'] else self.norm
        self.bottleneck_lstm2 = BottleneckLSTM(input_channels=config['bottleneck_lstm2']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm2']['out_channels'],
                                               norm=lstm_norm)

        # IR block 4 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block4 = self._make_ir_block(in_channels=config['ir_block4']['in_channels'],
                                             out_channels=config['ir_block4']['out_channels'],
                                             num_block=config['ir_block4']['num_block'],
                                             expansion_factor=config['ir_block4']['expansion_factor'],
                                             stride=config['ir_block4']['stride'], use_norm=config['ir_block4']['use_norm'])


    def forward(self, x):
        """
        @param x: input tensor for timestep t with shape (B, C, H, W)
        @return: list of features maps and hidden states (spatio-temporal features)
        """
        # Extracts spatial information
        x = self.in_conv(x)
        x = self.ir_block1(x)
        x = self.ir_block2(x)
        # Extract spatial and temporal representation at a first scale + update hidden states and cell states
        self.h_list[0], self.c_list[0] = self.bottleneck_lstm1(x, self.h_list[0], self.c_list[0])
        # Use last hidden state as input for the next convolutional layer
        st_features_lstm1 = self.h_list[0]
        x = self.ir_block3(st_features_lstm1)
        # Extract spatial and temporal representation at a second scale + update hidden states and cell states
        self.h_list[1], self.c_list[1] = self.bottleneck_lstm2(x, self.h_list[1], self.c_list[1])
        # Use last hidden state as input for the next convolutional layer
        st_features_lstm2 = self.h_list[1]
        st_features_backbone = self.ir_block4(st_features_lstm2)

        return st_features_backbone, st_features_lstm2, st_features_lstm1

    def _make_ir_block(self, in_channels, out_channels, num_block, expansion_factor, stride, use_norm):
        """
        Build an Inverted Residual bottleneck block
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param num_block: number of IR layer in the block
        @param expansion_factor: expansion factor of each IR layer
        @param stride: stride of the first convolution
        @return a torch.nn.Sequential layer
        """
        if use_norm:
            norm = self.norm
        else:
            norm = None
        layers = [InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                   expansion_factor=expansion_factor, norm=norm)]
        for i in range(1, num_block):
            layers.append(InvertedResidual(in_channels=out_channels, out_channels=out_channels, stride=1,
                                           expansion_factor=expansion_factor,  norm=norm))
        return nn.Sequential(*layers)

    def __init_hidden__(self):
        """
        Init hidden states and cell states list
        """
        # List of 2 hidden/cell states as we use 2 Bottleneck LSTM. The initialisation is done inside a Bottleneck LSTM cell.
        self.h_list = [None, None]
        self.c_list = [None, None]


class RecordDecoder(nn.Module):
    def __init__(self, config, n_class, norm_decoder="layer"):
        """
        RECurrent Online object detectOR (RECORD) decoder.

        @param config: config list to build the decoder
        @param n_class: number of output class
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param round_nearest: Round the number of channels in each layer to be a multiple of this number
        @param norm_decoder: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordDecoder, self).__init__()

        # Set the number of classes as the number of output channel of the last convolution
        config['conv_head2']['out_channels'] = n_class

        self.up_conv1 = nn.ConvTranspose2d(in_channels=config['conv_transpose1']['in_channels'],
                                           out_channels=config['conv_transpose1']['out_channels'],
                                           kernel_size=config['conv_transpose1']['kernel_size'],
                                           stride=config['conv_transpose1']['stride'],
                                           output_padding=config['conv_transpose1']['output_padding'],
                                           padding=config['conv_transpose1']['padding'])
        # Evaluate the sum of channels of the # channels of up_conv1 and # channels of the last hidden states of second
        # LSTM for the skip connection
        conv_norm = None if not config['conv_skip1']['use_norm'] else norm_decoder
        self.conv_skip_connection1 = InvertedResidual(in_channels=config['conv_skip1']['in_channels'],
                                                      out_channels=config['conv_skip1']['out_channels'],
                                                      expansion_factor=config['conv_skip1']['expansion_factor'],
                                                      stride=config['conv_skip1']['stride'],
                                                      norm=conv_norm)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=config['conv_transpose2']['in_channels'],
                                           out_channels=config['conv_transpose2']['out_channels'],
                                           kernel_size=config['conv_transpose2']['kernel_size'],
                                           stride=config['conv_transpose2']['stride'],
                                           output_padding=config['conv_transpose2']['output_padding'],
                                           padding=config['conv_transpose2']['padding'])
        # Evaluate the sum of channels of the # channels of up_conv2 and # channels of the last hidden states of first
        # LSTM for the skip connection
        conv_norm = None if not config['conv_skip2']['use_norm'] else norm_decoder
        self.conv_skip_connection2 = InvertedResidual(in_channels=config['conv_skip2']['in_channels'],
                                                      out_channels=config['conv_skip2']['out_channels'],
                                                      expansion_factor=config['conv_skip2']['expansion_factor'],
                                                      stride=config['conv_skip2']['stride'],
                                                      norm=conv_norm)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=config['conv_transpose3']['in_channels'],
                                           out_channels=config['conv_transpose3']['out_channels'],
                                           kernel_size=config['conv_transpose3']['kernel_size'],
                                           stride=config['conv_transpose3']['stride'],
                                           output_padding=config['conv_transpose3']['output_padding'],
                                           padding=config['conv_transpose3']['padding'])

        conv_norm = None if not config['conv_skip3']['use_norm'] else norm_decoder
        self.conv_skip_connection3 = InvertedResidual(in_channels=config['conv_skip3']['in_channels'],
                                                      out_channels=config['conv_skip3']['out_channels'],
                                                      expansion_factor=config['conv_skip3']['expansion_factor'],
                                                      stride=config['conv_skip3']['stride'],
                                                      norm=conv_norm)

        conv_norm = None if not config['conv_head1']['use_norm'] else norm_decoder
        self.conv_head1 = Conv3x3ReLUNorm(in_channels=config['conv_head1']['in_channels'],
                            out_channels=config['conv_head1']['out_channels'],
                            stride=config['conv_head1']['stride'], norm=conv_norm)
        self.conv_head2 = nn.Conv2d(in_channels=config['conv_head2']['in_channels'],
                                    out_channels=config['conv_head2']['out_channels'],
                                    kernel_size=config['conv_head2']['kernel_size'],
                                    stride=config['conv_head2']['stride'], padding=config['conv_head2']['padding'])

    def forward(self, st_features_backbone, st_features_lstm2, st_features_lstm1):
        """
        Forward pass RECORD decoder
        @param st_features_backbone: Last features map
        @param st_features_lstm2: Spatio-temporal features map from the second Bottleneck LSTM
        @param st_features_lstm1: Spatio-temporal features map from the first Bottleneck LSTM
        @return: ConfMap prediction (B, n_class, H, W)
        """
        # Spatio-temporal skip connection 1
        skip_connection1_out = torch.cat((self.up_conv1(st_features_backbone), st_features_lstm2), dim=1)
        x = self.conv_skip_connection1(skip_connection1_out)

        # Spatio-temporal skip connection 2
        skip_connection2_out = torch.cat((self.up_conv2(x), st_features_lstm1), dim=1)
        x = self.conv_skip_connection2(skip_connection2_out)

        x = self.up_conv3(x)
        x = self.conv_skip_connection3(x)

        x = self.conv_head1(x)
        x = self.conv_head2(x)
        return x


class Conv3x3ReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm='layer'):
        """
        Conv 3x3 + LayerNorm + LeakyReLU activation function module
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: LayerNorm). Set to None to disable normalisation.
        """
        super(Conv3x3ReLUNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=stride)
        self.acti = nn.LeakyReLU(inplace=True)
        
        if norm is not None:
            self.norm = nn.GroupNorm(1, out_channels)  
        else: 
            self.norm = None
        
    def forward(self, x):
        """
        Forward pass Conv3x3ReLUNorm module
        @param x: input tensor with shape (B, Cin, H, W)
        @return: output tensor with shape (B, Cout, H/s, W/s)
        """
        x = self.conv(x)
        x = self.acti(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, norm='layer'):
        """
        Modified MobileNetV2 Inverted Residual bottleneck layer with layer norm and
        LeakyReLU activation function.
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param expansion_factor: round the number of channels in each layer to be a multiple of this number
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: LayerNorm). Set to None to disable normalisation.
        """
        super(InvertedResidual, self).__init__()
        hidden_dim = round(in_channels * expansion_factor)
        self.identity = stride == 1 and in_channels == out_channels

        if norm is not None:
            n_group = 1
            norm_op = nn.GroupNorm
        else:
            norm_op = None

        if expansion_factor == 1:
            self.conv = nn.Sequential(
                #dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                        stride=1, padding=1, groups=hidden_dim),

                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                #pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                        stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=out_channels)
            )
        else:
            self.conv = nn.Sequential(
                #pw
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                        stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                #dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                        stride=stride, padding=1, groups=hidden_dim),
                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                #pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                        stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=out_channels)
            )

    def forward(self, x):
        """
        InvertedResidual bottleneck block forward pass
        @param x: input tensor with shape (B, Cin, H, W)
        @return: output tensor with shape (B, Cout, H/s, W/s)
        """
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BottleneckLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, norm='layer'):
        """
        From: https://github.com/vikrant7/mobile-vod-bottleneck-lstm/blob/master/network/mvod_bottleneck_lstm1.py
        Creates a bottleneck LSTM cell
        @param input_channels: number of input channels
        @param hidden_channels: number of hidden channels
        @param kernel_size: size of the kernel for convolutions (gates)
        @param norm: normalisation to use on output gates (default: LayerNorm) - Other normalisation not implemented yet
        """
        super(BottleneckLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = norm

        self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=kernel_size,
                           groups=self.input_channels, stride=1, padding=1)
        self.Wy = nn.Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
        self.Wi = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size, 1, 1,
                            groups=self.hidden_channels, bias=False)
        self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        if norm is not None:
            if norm == 'layer':
                self.norm_wbi = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbf = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbc = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbo = nn.GroupNorm(1, self.hidden_channels)

        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialized bias of the cell (default to 1)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, x, h, c):
        """
        Forward pass Bottleneck LSTM cell
        @param x: input tensor with shape (B, C, H, W)
        @param h: hidden states
        @param c: cell states
        @return: new hidden states and new cell states
        """
        x = self.W(x)
        # Concat "gate": concatenate input and hidden layers
        y = torch.cat((x, h),1) 
        # Bottleneck gate: reduce to hidden layer size
        i = self.Wy(y) 
        b = self.Wi(i)	# depth wise 3*3
        
        # Input gate
        if self.norm is not None:
            ci = self.sigmoid(self.norm_wbi(self.Wbi(b)))
        else:
            ci = self.sigmoid(self.Wbi(b))

        # Forget gate
        if self.norm is not None:
            cf = self.sigmoid(self.norm_wbf(self.Wbf(b)))
        else:
            cf = self.sigmoid(self.Wbf(b))

        # Multiply forget gate with cell state + add output of
        # input gate multiplied by output of the conv after bottleneck gate
        if self.norm is not None:
            cc = cf * c + ci * self.relu(self.norm_wbc(self.Wbc(b)))
        else:
            cc = cf * c + ci * self.relu(self.Wbc(b))

        # Output gate
        if self.norm is not None:
            co = self.sigmoid(self.norm_wbo(self.Wbo(b)))
        else:
            co = self.sigmoid(self.Wbo(b))

        ch = co * self.relu(cc)
        return ch, cc

    @staticmethod
    def init_hidden(batch_size, hidden, shape):
        # Mandatory to specify cuda here as Pytorch Lightning doesn't do it automatically for new tensors
        if torch.cuda.is_available():
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
        else:
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        return h_init, c_init


class BottleneckLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, norm='layer'):
        """
        Single layer Bottleneck LSTM cell
        @param input_channels: number of input channels of the cell
        @param hidden_channels: number of hidden channels of the cell
        @param norm: normalisation to use (default: LayerNorm) - Other normalisation are not implemented yet.
        """
        super(BottleneckLSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)

        self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels, norm=norm)

    def forward(self, inputs, h, c):
        """
        Forward pass Bottleneck LSTM layer
        If stateful LSTM h and c must be None. Else they must be Tensor.
        @param inputs: input tensor
        @param h: hidden states (if None, they are automatically initialised)
        @param c: cell states (if None, they are automatically initialised)
        @return: new hidden and cell states
        """
        if h is None and c is None:
            h, c = self.cell.init_hidden(batch_size=inputs.shape[0], hidden=self.hidden_channels,
                                         shape=(inputs.shape[-2], inputs.shape[-1]))
        new_h, new_c = self.cell(inputs, h, c)
        return new_h, new_c
    

class RecordEncoderNoLstm(nn.Module):
    def __init__(self, in_channels, config, norm='layer'):
        """
        RECurrent Online object detectOR (RECORD) features extractor.
        @param in_channels: number of input channels (default: 8)
        @param config: number of input channels per block
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordEncoderNoLstm, self).__init__()
        self.norm = norm
        # Set the number of input channels in the configuration file
        config['in_conv']['in_channels'] = in_channels

        # config_tmp = **config['in_conv'] --> dict with arguments

        # Input convolution (expands the number of input channels)
        self.in_conv = Conv3x3ReLUNorm(in_channels=config['in_conv']['in_channels'],
                                       out_channels=config['in_conv']['out_channels'],
                                       stride=config['in_conv']['stride'], norm=norm)

        # IR block 1 (acts as a bottleneck)
        self.ir_block1 = self._make_ir_block(in_channels=config['ir_block1']['in_channels'],
                                             out_channels=config['ir_block1']['out_channels'],
                                             num_block=config['ir_block1']['num_block'],
                                             expansion_factor=config['ir_block1']['expansion_factor'],
                                             stride=config['ir_block1']['stride'], use_norm=config['ir_block1']['use_norm'])

        # IR block 2 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block2 = self._make_ir_block(in_channels=config['ir_block2']['in_channels'],
                                             out_channels=config['ir_block2']['out_channels'],
                                             num_block=config['ir_block2']['num_block'],
                                             expansion_factor=config['ir_block2']['expansion_factor'],
                                             stride=config['ir_block2']['stride'], use_norm=config['ir_block2']['use_norm'])

        # IR block 3 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block3 = self._make_ir_block(in_channels=config['ir_block3']['in_channels'],
                                             out_channels=config['ir_block3']['out_channels'],
                                             num_block=config['ir_block3']['num_block'],
                                             expansion_factor=config['ir_block3']['expansion_factor'],
                                             stride=config['ir_block3']['stride'], use_norm=config['ir_block3']['use_norm'])


        # IR block 4 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block4 = self._make_ir_block(in_channels=config['ir_block4']['in_channels'],
                                             out_channels=config['ir_block4']['out_channels'],
                                             num_block=config['ir_block4']['num_block'],
                                             expansion_factor=config['ir_block4']['expansion_factor'],
                                             stride=config['ir_block4']['stride'], use_norm=config['ir_block4']['use_norm'])


    def forward(self, x):
        """
        @param x: input tensor for timestep t with shape (B, C, H, W)
        @return: list of features maps and hidden states (spatio-temporal features)
        """
        # Extracts spatial information
        x = self.in_conv(x)
        x = self.ir_block1(x)
        x1 = self.ir_block2(x)
        x2 = self.ir_block3(x1)
        x3 = self.ir_block4(x2)

        return x3, x2, x1

    def _make_ir_block(self, in_channels, out_channels, num_block, expansion_factor, stride, use_norm):
        """
        Build an Inverted Residual bottleneck block
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param num_block: number of IR layer in the block
        @param expansion_factor: expansion factor of each IR layer
        @param stride: stride of the first convolution
        @return a torch.nn.Sequential layer
        """
        if use_norm:
            norm = self.norm
        else:
            norm = None
        layers = [InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                   expansion_factor=expansion_factor, norm=norm)]
        for i in range(1, num_block):
            layers.append(InvertedResidual(in_channels=out_channels, out_channels=out_channels, stride=1,
                                           expansion_factor=expansion_factor,  norm=norm))
        return nn.Sequential(*layers)
    

def soft_dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
                   global_weight: float = 1.) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(torch.pow(input_soft, 2) + torch.pow(target_one_hot, 2), dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return global_weight*torch.mean(-dice_score + 1.)


class SoftDiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(SoftDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.global_weight = global_weight

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(input, target, self.eps, self.global_weight)
    

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps