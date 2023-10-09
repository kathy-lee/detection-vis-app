import torch
import torch.nn as nn
import torch.nn.functional as F




def mish_activation(x):
    """ Mish Activation Function """
    return x * torch.tanh(F.softplus(x))  


class Convolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, 
                 use_activation=True, use_bias=True, bn=True):
        """
        2D convolutional layer in this research 

        Args:
            in_channels: Input channels, int
            out_channels: Output channels, int
            kernel_size: Kernel size, int
            stride: Stride, int or tuple
            padding: Padding, either 'same' or 'valid'
            activation: Activation function, either 'relu' or 'mish'
        """
        super(Convolution2D, self).__init__()

        if padding == "same":
            padding = kernel_size // 2  # This provides "same" padding for odd kernel sizes
        elif padding == "valid":
            padding = 0
        else:
            raise ValueError("Expected padding to be 'same' or 'valid', but got {}".format(padding))

        # Regularization is typically done using weight decay in the optimizer in PyTorch
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.use_activation = use_activation
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        
        if self.bn:
            x = self.bn(x)
            
        if self.use_activation:
            if self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "mish":
                x = mish_activation(x)
            else:
                raise ValueError("Unsupported activation function")       
        return x


class MaxPooling2D(nn.Module):
    def __init__(self, pool_size=(2,2), strides=(2,2), padding="same"):
        """
        Max pooling layer for 2D tensors
        
        Args:
            pool_size: Size of the max pooling window, tuple
            strides: Stride of the max pooling window, tuple
            padding: Padding, either 'same' or 'valid'
        """
        super(MaxPooling2D, self).__init__()

        # Adjusting padding to be compatible with 'same' and 'valid' padding modes from TensorFlow
        if padding == 'same':
            self.padding = ((pool_size[0]-1)//2, (pool_size[1]-1)//2)
        else:  # 'valid'
            self.padding = (0, 0)

        self.pool_size = pool_size
        self.strides = strides

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.strides, padding=self.padding)
    

class BasicResidualBlock(nn.Module):
    def __init__(self, input_channels, channel_expansion, strides=(1, 1), use_bias=False):
        super(BasicResidualBlock, self).__init__()

        self.channel_expansion = channel_expansion
        self.strides = strides
        self.use_bias = use_bias

        # First convolution
        self.conv1 = Convolution2D(input_channels, input_channels, 3, strides, padding='same',
                                   activation='relu', use_bias=use_bias, bn=True)
        
        # Second convolution
        self.conv2 = Convolution2D(input_channels, int(input_channels * channel_expansion), 3, (1, 1), 
                                   padding='same', activation='relu', use_bias=use_bias, bn=True)
        
        # Third convolution
        self.conv3 = Convolution2D(int(input_channels * channel_expansion), int(input_channels * channel_expansion),
                                   1, (1, 1), padding='same', activation='relu', use_bias=use_bias, bn=True)
        
        # Shortcut
        self.shortcut = Convolution2D(input_channels, int(input_channels * channel_expansion), 3, 
                                      strides, padding='same', activation='relu', use_bias=use_bias, bn=True)

    def forward(self, x):
        out1 = self.conv1(x)
        #print(f"1st conv passed. input:{x.shape}, output: {out1.shape}")
        out2 = self.conv2(out1)
        #print(f"2nd conv passed. output:{out2.shape}")
        out3 = self.conv3(out2)
        #print(f"3rd conv passed. output:{out3.shape}")

        # Decide which path to take for the shortcut
        if any(val != 1 for val in self.strides) or self.channel_expansion != 1:
            shortcut_out = self.shortcut(x)
        else:
            shortcut_out = x

        # Combining the outputs
        out = out3 + shortcut_out
        return out


class RepeatBlock(nn.Module):
    def __init__(self, input_channels, repeat_times, all_strides=None, all_expansions=None, feature_maps_downsample=False):
        super(RepeatBlock, self).__init__()
        
        # Ensure correct parameter formats and values
        if all_strides is not None and all_expansions is not None:
            assert (isinstance(all_strides, (tuple, list)))
            assert (isinstance(all_expansions, (tuple, list)))
            assert len(all_strides) == repeat_times
            assert len(all_expansions) == repeat_times
        elif all_strides is None and all_expansions is None:
            all_strides, all_expansions = [], []
            for i in range(repeat_times):
                all_strides.append(1)
                all_expansions.append(0.5 if i % 2 == 0 else 2)

        # Build the repeated blocks
        blocks = []
        for i in range(repeat_times):
            strides = (all_strides[i], all_strides[i])
            expansion = all_expansions[i]
            blocks.append(BasicResidualBlock(input_channels, expansion, strides, use_bias=True))
            input_channels = int(input_channels * expansion)
        self.blocks = nn.Sequential(*blocks)
        
        # Optional MaxPooling2D layer
        self.feature_maps_downsample = feature_maps_downsample
        if feature_maps_downsample:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.blocks(x)
        #print(f"repeat block passed. output shape: {x.shape}")
        if self.feature_maps_downsample:
            x = self.maxpool(x)
        return x
    

class RadarResNet3D(nn.Module):
    def __init__(self, input_channels):
        super(RadarResNet3D, self).__init__()

        # Configuration parameters
        self.block_repeat_times = [2, 4, 8, 16]
        self.channels_upsample = [False, False, True, True]
        self.feature_mp_downsample = [True, True, True, True]
        
        # Create the repeated blocks based on the configuration
        self.blocks = nn.ModuleList()
        for i, repeat_times in enumerate(self.block_repeat_times):
            if repeat_times != 1:
                all_strides = [1, 1] * int(repeat_times/2)
                all_expansions = [1, 1] * int(repeat_times/2)
            else:
                all_strides = [1] * int(repeat_times)
                all_expansions = [1] * int(repeat_times)
            if self.channels_upsample[i]:
                all_expansions[-1] *= 2

            feature_maps_downsample = self.feature_mp_downsample[i]
            block = RepeatBlock(input_channels, repeat_times, all_strides, all_expansions, feature_maps_downsample)
            input_channels = int(input_channels * all_expansions[-1])
            self.blocks.append(block)

    def forward(self, x):
        feature_stages = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > len(self.block_repeat_times) - 4:
                feature_stages.append(x)
            #print(f"block {i} passed. output shape : {x.shape}")

        # NOTE: since we are doing one-level output, only last level is used
        features = feature_stages[-1]
        #print(f"RadarResNet3D passed....feature output: {features.shape}")
        return features
    


class YoloHead(nn.Module):
    def __init__(self, num_anchors_layer, num_class, input_channel, last_channel):
        super(YoloHead, self).__init__()

        self.num_anchors_layer = num_anchors_layer
        self.num_class = num_class
        self.last_channel = last_channel   

        # Calculate final output channels
        self.final_output_channels = self.last_channel * self.num_anchors_layer * (self.num_class + 7)

        # Convolution Layers
        # First Conv layer
        self.conv1 = Convolution2D(input_channel, input_channel * 2, 
                                   kernel_size=3, stride=(1,1), padding='same', 
                                   activation='relu', use_activation=True, use_bias=True, bn=True)
        
        # Second Conv layer
        self.conv2 = Convolution2D(input_channel * 2, self.final_output_channels, 
                                   kernel_size=1, stride=(1,1), padding='same', 
                                   activation=None, use_activation=False, use_bias=True, bn=False)

    def forward(self, feature_map):
        # Get the required shape
        batch_size = feature_map.size(0)
        height = feature_map.size(2)
        width = feature_map.size(3)
        final_output_reshape = (batch_size, self.last_channel, self.num_anchors_layer * (self.num_class + 7), height, width)
        
        # First convolution
        conv = self.conv1(feature_map)
        #print(f"1st conv in YoloHead passed. Output: {conv.shape}")
        # Second convolution
        conv = self.conv2(conv)
        #print(f"2nd conv in YoloHead passed. Output: {conv.shape}")
        # Reshape operation
        output = conv.view(final_output_reshape)
        #print(f"final output : {output.shape}")
        return output
