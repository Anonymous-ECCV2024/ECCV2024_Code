from importlib.util import module_for_loader
from timm.models import create_model
import torch.nn as nn

def mixer_rm_modules(model):
    num_blocks = len(model.blocks)
    
    rm_modules = [(model.blocks[n].mlp_tokens.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_tokens.fc2, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_channels.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_channels.fc2, 'weight') for n in range(num_blocks)]
    
    return tuple(rm_modules)

def vit_rm_modules(model):
    num_blocks = len(model.blocks)

    rm_modules = [(model.blocks[n].attn.qkv, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].attn.proj, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp.fc2, 'weight') for n in range(num_blocks)]
    
    return tuple(rm_modules)

def pool_rm_modules(model):
    rm_modules = []

    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[0]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[0]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[2]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[2]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[4]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[4]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[6]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[6]]

    return tuple(rm_modules)


def resnet_rm_modules(model):
    num_blocks = len(model.layer1)
    rm_modules = []

    rm_modules += [(model.conv1, 'weight')]
    rm_modules += [(model.bn1, 'weight')]
    
    rm_modules += [(module.conv1, 'weight') for module in model.layer1]
    rm_modules += [(module.bn1, 'weight') for module in model.layer1]
    rm_modules += [(module.conv2, 'weight') for module in model.layer1]
    rm_modules += [(module.bn2, 'weight') for module in model.layer1]
    rm_modules += [(module.conv3, 'weight') for module in model.layer1]
    rm_modules += [(module.bn3, 'weight') for module in model.layer1]
    rm_modules += [(model.layer1[0].downsample[0], 'weight')]
    rm_modules += [(model.layer1[0].downsample[1], 'weight')]
    
    rm_modules += [(module.conv1, 'weight') for module in model.layer2]
    rm_modules += [(module.bn1, 'weight') for module in model.layer2]
    rm_modules += [(module.conv2, 'weight') for module in model.layer2]
    rm_modules += [(module.bn2, 'weight') for module in model.layer2]
    rm_modules += [(module.conv3, 'weight') for module in model.layer2]
    rm_modules += [(module.bn3, 'weight') for module in model.layer2]
    rm_modules += [(model.layer2[0].downsample[0], 'weight')]
    rm_modules += [(model.layer2[0].downsample[1], 'weight')]
    
    rm_modules += [(module.conv1, 'weight') for module in model.layer3]
    rm_modules += [(module.bn1, 'weight') for module in model.layer3]
    rm_modules += [(module.conv2, 'weight') for module in model.layer3]
    rm_modules += [(module.bn2, 'weight') for module in model.layer3]
    rm_modules += [(module.conv3, 'weight') for module in model.layer3]
    rm_modules += [(module.bn3, 'weight') for module in model.layer3]
    rm_modules += [(model.layer3[0].downsample[0], 'weight')]
    rm_modules += [(model.layer3[0].downsample[1], 'weight')]

    rm_modules += [(module.conv1, 'weight') for module in model.layer4]
    rm_modules += [(module.bn1, 'weight') for module in model.layer4]
    rm_modules += [(module.conv2, 'weight') for module in model.layer4]
    rm_modules += [(module.bn2, 'weight') for module in model.layer4]
    rm_modules += [(module.conv3, 'weight') for module in model.layer4]
    rm_modules += [(module.bn3, 'weight') for module in model.layer4]
    rm_modules += [(model.layer4[0].downsample[0], 'weight')]
    rm_modules += [(model.layer4[0].downsample[1], 'weight')]

    
    return tuple(rm_modules)

def vgg_rm_modules(model):
    rm_modules = []
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            rm_modules += [(layer, 'weight')]
    
    rm_modules += [(model.pre_logits.fc1,'weight')]
    rm_modules += [(model.pre_logits.fc2,'weight')]
    
    return tuple(rm_modules)
