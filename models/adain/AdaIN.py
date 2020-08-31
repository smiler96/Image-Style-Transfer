import torch.nn as nn
import torch
import torchvision.models as models
from models.common.core import conv_default, calc_mean_std
from losses.loss import content_loss, style_loss
from torch.utils.tensorboard import SummaryWriter

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class AdaIN_Bottle(nn.Module):
    def __init__(self):
        super(AdaIN_Bottle, self).__init__()

    def forward(self, x, style):
        x = adaptive_instance_normalization(x, style)
        return x

class AdaIN_Net(nn.Module):
    def __init__(self):
        super(AdaIN_Net, self).__init__()

        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('E:/Style-Transfer-Zoo/models/pretrained_weights/vgg19-dcbb9e9d.pth'))
        enc_layers = nn.Sequential(*list(vgg19.children())[0][:37])
        # enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:2])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[2:7])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[7:12])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[12:21])  # relu3_1 -> relu4_1

        self.adain = AdaIN_Bottle()
        # self.encoder = None
        self.decoder = nn.Sequential(*[
            conv_default(512, 256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            conv_default(256, 256),
            nn.ReLU(),
            conv_default(256, 256),
            nn.ReLU(),
            conv_default(256, 256),
            nn.ReLU(),
            conv_default(256, 128),
            nn.ReLU(),
            nn.Upsample(scale_
