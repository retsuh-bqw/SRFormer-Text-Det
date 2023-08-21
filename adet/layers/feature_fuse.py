import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels = 256, inter_channels = 64 , out_features_num=4, attention_type='scale_channel_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        inner_channels = inter_channels
        self.out_features_num = out_features_num
        bias = False

        self.output_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=bias),
                            nn.GroupNorm(32, in_channels)) for i in range(4)])

        self.lateral_conv_4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.linear_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(ori_channels, in_channels, 1, bias=bias),
                            nn.GroupNorm(32, in_channels)) for ori_channels in [256, 512, 1024, 2048]])
        
        for i in range(4):
            self.output_convs[i].apply(self._initialize_weights)
        self.lateral_conv_4.apply(self._initialize_weights)
        self.lateral_conv_3.apply(self._initialize_weights)
        self.lateral_conv_2.apply(self._initialize_weights)




    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('GroupNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def forward(self, encoder_features, bk_feature):
        c2, c3, c4, c5 = encoder_features

        # Basic FPN
        out4 = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_4(c4) \
               + self.linear_proj[2](F.interpolate(bk_feature[2], size=c4.shape[-2:], mode="bilinear", align_corners=False))
        out3 = F.interpolate(out4, size=c3.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_3(c3) \
               + self.linear_proj[1](F.interpolate(bk_feature[1], size=c3.shape[-2:], mode="bilinear", align_corners=False))
        out2 = F.interpolate(out3, size=c2.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_2(c2) \
               + self.linear_proj[0](F.interpolate(bk_feature[0], size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p5 = self.output_convs[0](c5 + self.linear_proj[3](F.interpolate(bk_feature[3], size=c5.shape[-2:], mode="bilinear", align_corners=False)))   
        p4 = self.output_convs[1](out4) 
        p3 = self.output_convs[2](out3) 
        p2 = self.output_convs[3](out2) 

        multiscale_feature = [p2, p3, p4, p5]


        return multiscale_feature