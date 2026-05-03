from munch import Munch

import numpy as np
import torch
import torch.nn as nn

from src.models.modules import MLPBlock,MaskDropout, prepare_VB_config, VBModule, get_layer
from src.models.cvb_dp_defense import build_dp_defense_config, sanitize_gradients_, arm_cvb_epsilon_defense_

def build_CNN(architecture, config):    
    conv_activation_function = config.model.conv_activation_function
    conv_regularization_layer = config.model.regularization_layer
    mlp_activation_function = config.model.mlp_activation_function
    mlp_regularization_layer = config.model.mlp_regularization_layer
    pooling_layer = config.model.pooling_layer
    if architecture == 'CNN':
        model = CNN(config.data.shape, config.num_classes, channels=config.model.channels, kernels=config.model.kernels, stride=config.model.stride, padding=config.model.padding, widths=config.model.widths, pooling_layer=pooling_layer, pooling_positions=config.model.pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, headless=config.model.headless)
    elif architecture == 'PlainCNN':
        model = PlainCNN(
            data_shape=config.data.shape,
            num_classes=config.num_classes,
            channels=getattr(config.model, 'channels', [32, 64, 128]),
            kernels=getattr(config.model, 'kernels', [3, 3, 3]),
            stride=getattr(config.model, 'stride', 1),
            padding=1 if getattr(config.model, 'padding', 'same') == 'same' else getattr(config.model, 'padding', 1),
            pooling_positions=getattr(config.model, 'pooling_positions', [0, 1, 2]),
            hidden_dim=getattr(config.model, 'hidden_dim', 256),
            headless=getattr(config.model, 'headless', False))
    elif architecture =='VBCNN':
        model = VBCNN(config.data.shape, config.num_classes, channels=config.model.channels, kernels=config.model.kernels, stride=config.model.stride, padding=config.model.padding, widths=config.model.widths, pooling_layer=pooling_layer, pooling_positions=config.model.pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, headless=config.model.headless, VB_config=prepare_VB_config(config.model.VB), defense_config=getattr(config.model, 'privacy_defense', None))
    else:
        raise ValueError(f'The model architecture {architecture} is not implemented yet..')
    return model

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', regularization_layer=None, activation_function=None, pooling_layer=None, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.regularization_layer = get_layer(regularization_layer.name, regularization_layer.parameters, channels = out_channels) if regularization_layer is not None else None
        self.activation_function = get_layer(activation_function.name, activation_function.parameters) if activation_function is not None else None
        self.pooling_layer = pooling_layer
        self.flatten_layer = nn.Flatten() if flatten else None

    def forward(self, x):
        x = self.conv(x)
        if self.regularization_layer is not None:
            x = self.regularization_layer(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x)
        if self.flatten_layer is not None:
            x = self.flatten_layer(x)
        return x

class CNN(nn.Module):
    def __init__(self, data_shape=(3,32,32), num_classes=10, channels=[64,128,128,256,256,256,256,256,256], kernels=[3,3,3,3,3,3,3,3,3], stride = 1, padding = 'same', widths=[], pooling_layer=nn.MaxPool2d(3), pooling_positions=[5], conv_regularization_layer=None, conv_activation_function=None, mlp_regularization_layer=None, mlp_activation_function=None, headless=False, **kwargs):
        super().__init__()
        if isinstance(pooling_layer, Munch):
            pooling_layer = get_layer(pooling_layer.name, pooling_layer.parameters)
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels)):
            in_channels = data_shape[0] if i == 0 else channels[i-1]
            out_channels = channels[i]
            kernel_size = kernels[i]
            pool = pooling_layer if i in pooling_positions else None
            #flatten = i == (len(channels)-1) # flatten if its the last convolutional block
            flatten = False
            self.conv_layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding, regularization_layer=conv_regularization_layer, activation_function=conv_activation_function, pooling_layer=pool, flatten=flatten))
                
        self.mlp_layers = nn.ModuleList()
        num_features = np.prod(self.get_layer_shape(len(channels)-1, data_shape))
        # print('num_features', num_features, self.get_layer_shape(len(channels)-1, data_shape))
        for i in range(len(widths)):
            in_channels = num_features if i == 0 else widths[i-1]
            out_channels = widths[i]
            self.mlp_layers.append(MLPBlock(in_channels, out_channels, mlp_regularization_layer, mlp_activation_function))    
        last_in = num_features if len(widths) == 0 else widths[-1]
        if not headless:
            self.mlp_layers.append(MLPBlock(last_in, num_classes, None, None)) #last layer

    def get_layer_shape(self, layer_idx, in_shape):
        with torch.no_grad():
            x = torch.rand(2, *in_shape)
            i = 0
            for layer in self.conv_layers:
                x = layer(x)
                if i == layer_idx:
                    return x.shape[1:]
                i += 1
            x = x.flatten(start_dim=1)
            for layer in self.mlp_layers:
                x = layer(x)
                if i == layer_idx:
                    return x.shape[1:]
                i += 1


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.mlp_layers:
            x = layer(x)
        return x
class PlainCNN(nn.Module):
    def __init__(
        self,
        data_shape=(3, 32, 32),
        num_classes=10,
        channels=[32, 64, 128],
        kernels=[3, 3, 3],
        stride=1,
        padding=1,
        pooling_positions=[0, 1, 2],
        hidden_dim=256,
        headless=False,
        **kwargs
    ):
        super().__init__()

        assert len(channels) == len(kernels), "channels and kernels must have the same length"

        self.features = nn.ModuleList()
        in_channels = data_shape[0]

        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernels)):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
            ]
            if i in pooling_positions:
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.append(nn.Sequential(*block))
            in_channels = out_channels

        with torch.no_grad():
            x = torch.zeros(1, *data_shape)
            for layer in self.features:
                x = layer(x)
            flattened_dim = x.flatten(start_dim=1).shape[1]

        self.classifier = nn.ModuleList()
        if not headless:
            self.classifier.append(nn.Linear(flattened_dim, hidden_dim))
            self.classifier.append(nn.ReLU(inplace=True))
            self.classifier.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.classifier:
            x = layer(x)
        return x

import torch.nn.functional as F

class VBCNN(VBModule, CNN):
    def __init__(self, data_shape=(3,32,32), num_classes=10, channels=[64,128,128,256,256,256,256,256,256], kernels=[3,3,3,3,3,3,3,3,3], stride = 1, padding = 'same', widths=[], pooling_layer=nn.MaxPool2d(3), pooling_positions=[5], conv_regularization_layer=None, conv_activation_function=None, mlp_regularization_layer=None, mlp_activation_function=None, VB_config=None, defense_config=None, **kwargs):
        super().__init__(data_shape=data_shape, num_classes=num_classes, channels=channels, kernels=kernels, stride=stride, padding=padding, widths=widths, pooling_layer=pooling_layer, pooling_positions=pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, **kwargs)
        self.input_VB = VB_config.VB_class(data_shape, VB_config.K, VB_config.beta, VB_config=VB_config) if -1 in VB_config.positions else None
        self.dropout_p = 0
        self.channels = channels
        self.width = widths
        for i in range(len(channels)+len(widths)): # CNN blocks, MLP blocks
            if i in VB_config.positions:
                self.VB_list.append(VB_config.VB_class(self.get_layer_shape(i, data_shape), VB_config.K, VB_config.beta, VB_config=VB_config))
            else:
                self.VB_list.append(None)
        self.VB_list.append(None) # Never have a VB after the final Layer
        total_dropouts = 1 + len(self.conv_layers) + 1 + len(self.mlp_layers)

        self.dropout_layers = nn.ModuleList([
            MaskDropout(self.dropout_p) for _ in range(total_dropouts)
        ])

        # Combined defense from CDIA paper: CVB + low-level DP on shared gradients.
        self.privacy_defense = build_dp_defense_config(defense_config)

    def configure_privacy_defense(self, defense_config):
        self.privacy_defense = build_dp_defense_config(defense_config)
        return self.privacy_defense

    def sanitize_shared_gradients(self, batch_size=None):
        """Call this after loss.backward() and before optimizer.step()/sharing gradients."""
        return sanitize_gradients_(self, self.privacy_defense, batch_size=batch_size)



    def forward(self, x):
        # print('IN', x.shape)
        drop_idx = 0
        if self.input_VB is not None:
            x = self.input_VB(x)
            # print('InputVB', x.shape)
        x = self.dropout_layers[drop_idx](x)
        drop_idx += 1
      
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            # print('CNN', i, x.shape)
            if self.VB_list[i] is not None:
                # If VBIA has injected a dummy_epsilon into this CVB module,
                # corrupt it with private noise the attacker cannot observe.
                # This fires only when fixed_epsilon is set (attacker closure)
                # and only in training mode, so normal inference is unaffected.
                arm_cvb_epsilon_defense_(self, self.privacy_defense)
                x = self.VB_list[i](x)
                # print('VB', x.shape)
            x = self.dropout_layers[drop_idx](x)
            drop_idx += 1
        
        x = x.flatten(start_dim=1)
        x = self.dropout_layers[drop_idx](x)
        drop_idx += 1
      

        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            # print('MLP', i, x.shape)
            if self.VB_list[i+len(self.conv_layers)] is not None:
                arm_cvb_epsilon_defense_(self, self.privacy_defense)
                x = self.VB_list[i+len(self.conv_layers)](x)          
                # print('VB', x.shape)
            x = self.dropout_layers[drop_idx](x)
            drop_idx += 1
           
        return x


    #Dropout tracking methods
    def track_dropout_mask(self, track=True):
        for module in self.modules():
            if isinstance(module, MaskDropout):
                module.track_mask = track

    def use_dropout_mask(self, use=True):
        for module in self.modules():
            if isinstance(module, MaskDropout):
                module.use_mask = use

    def get_dropout_layers(self, positions=None):
        all_dl_layers = [module for module in self.modules() if isinstance(module, MaskDropout)]
        if positions is not None:
            return [dl for i, dl in enumerate(all_dl_layers) if i in positions]
        return all_dl_layers