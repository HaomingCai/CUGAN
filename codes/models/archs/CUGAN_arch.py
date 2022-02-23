import functools
import torch.nn as nn
import models.archs.arch_util as arch_util



# Basic Block
class CResBlock(nn.Module):
    def __init__(self, nf=64, cond_dim=2):
        super(CResBlock, self).__init__()

        # Network Component
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.local_scale = nn.Linear(cond_dim, nf, bias=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)
        arch_util.initialize_weights([self.local_scale], 0.1)

    def forward(self, x):
        content, cond = x

        fea = self.conv1(content)
        out = self.conv2(self.act(fea))
        local_scale = self.local_scale(cond)

        return content + out * local_scale.view(-1, content.size()[1], 1, 1), cond



class CUGAN(nn.Module):
    def __init__(self, in_nc, out_nc, cond_dim, stages_blocks_num, stages_channels, downSample_Ksize):
        super(CUGAN, self).__init__()

        self.stage1_nb, self.stage2_nb, self.stage3_nb = stages_blocks_num[0], stages_blocks_num[1], stages_blocks_num[2]
        self.stage1_nf, self.stage2_nf, self.stage3_nf = stages_channels[0], stages_channels[1], stages_channels[2]
        self.DownSample_Ksize = downSample_Ksize
        self.cond_dim    = cond_dim


        # stage1 left
        self.conv_stage1_left = nn.Conv2d(in_nc, self.stage1_nf, kernel_size=3, stride=1, padding=1, bias=True)
        CResBlock_stage1_left = functools.partial(CResBlock, nf=self.stage1_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage1_left = arch_util.make_layer(CResBlock_stage1_left, self.stage1_nb)
        #
        # stage2 left
        self.conv_stage2_left = nn.Conv2d(self.stage1_nf, self.stage2_nf, kernel_size=self.DownSample_Ksize, stride=2, padding=0, bias=True)
        CResBlock_stage2_left = functools.partial(CResBlock, nf=self.stage2_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage2_left = arch_util.make_layer(CResBlock_stage2_left, self.stage2_nb)
        #
        # stage3 left
        self.conv_stage3_left = nn.Conv2d(self.stage2_nf, self.stage3_nf, kernel_size=self.DownSample_Ksize, stride=2, padding=0, bias=True)
        CResBlock_stage3_left = functools.partial(CResBlock, nf=self.stage3_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage3_left = arch_util.make_layer(CResBlock_stage3_left, self.stage3_nb)
        #
        #
        #
        # stage3 right
        CResBlock_stage3_right = functools.partial(CResBlock, nf=self.stage3_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage3_right = arch_util.make_layer(CResBlock_stage3_right, self.stage3_nb)
        self.conv_stage3_right = nn.ConvTranspose2d(self.stage3_nf, self.stage2_nf, kernel_size=2, stride=2, bias=True)
        arch_util.initialize_weights([self.conv_stage3_left, self.conv_stage3_right], 0.1)
        #
        #
        # stage2 right
        CResBlock_stage2_right = functools.partial(CResBlock, nf=self.stage2_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage2_right = arch_util.make_layer(CResBlock_stage2_right, self.stage2_nb)
        self.conv_stage2_right = nn.ConvTranspose2d(self.stage2_nf, self.stage1_nf, kernel_size=2, stride=2, bias=True)
        arch_util.initialize_weights([self.conv_stage2_left, self.conv_stage2_right], 0.1)
        #
        #
        # stage1 right
        CResBlock_stage1_right = functools.partial(CResBlock, nf=self.stage1_nf, cond_dim=self.cond_dim)
        self.CResBlocks_stage1_right = arch_util.make_layer(CResBlock_stage1_right, self.stage1_nb)
        self.conv_stage1_right = nn.Conv2d(self.stage1_nf, out_nc, kernel_size=3, stride=1, padding=1, bias=True)
        arch_util.initialize_weights([self.conv_stage1_left, self.conv_stage1_right], 0.1)

        # condition scale for stages
        self.stage3_scale = nn.Linear(self.cond_dim, self.stage2_nf)
        self.stage2_scale = nn.Linear(self.cond_dim, self.stage1_nf)
        self.stage1_scale = nn.Linear(self.cond_dim, out_nc)
        arch_util.initialize_weights([self.stage1_scale, self.stage2_scale, self.stage3_scale], 0.1)


    def forward(self, inputs):
        content, cond = inputs

        # stage1 left
        out = self.conv_stage1_left(content)
        stage1_left, _ = self.CResBlocks_stage1_left((out, cond))

        # stage2 left
        out = self.conv_stage2_left(stage1_left)
        stage2_left, _ = self.CResBlocks_stage2_left((out, cond))

        # stage3 left
        out = self.conv_stage3_left(stage2_left)
        stage3_left, _ = self.CResBlocks_stage3_left((out, cond))


        # stage3 right
        stage3_right, _ = self.CResBlocks_stage3_right((stage3_left, cond))
        stage3_right = self.conv_stage3_right(stage3_right)
        # stage 3 Modulated Scale Fusion
        stage3_scale = self.stage3_scale(cond).view(-1, stage3_right.size()[1], 1, 1)
        stage2_left = stage2_left + stage3_scale * stage3_right


        # stage2 right
        stage2_right, _ = self.CResBlocks_stage2_right((stage2_left, cond))
        stage2_right = self.conv_stage2_right(stage2_right)
        # stage 2 Modulated Scale Fusion
        stage2_scale = self.stage2_scale(cond).view(-1, stage2_right.size()[1], 1, 1)
        stage1_left = stage1_left + stage2_scale * stage2_right


        # stage1 right
        stage1_right, _ = self.CResBlocks_stage1_right((stage1_left, cond))
        stage1_right = self.conv_stage1_right(stage1_right)
        # stage 1 Modulated Scale Fusion
        stage1_scale = self.stage1_scale(cond).view(-1, stage1_right.size()[1], 1, 1)
        final = content + stage1_scale * stage1_right

        return final
