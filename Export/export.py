from torch.nn import init
import torch
# import torch.nn.functional as F
# from utils.objective import GANLoss, MSELoss, AngularLoss
# from torch.nn.parallel import gather, parallel_apply, replicate
from torch import nn
from nets.Student_Generator import Student_G
# from nets.Teacher_Generator import Teacher_G
# import time
from torchvision.utils import save_image
import os
from nets.export_Gen import Expot

def transfer_ConV2d(mA, mB):
    a, b, c, d = mB.weight.data.shape
    mB.weight.data = mA.weight.data[:a, :b, :c, :d]
    if mB.bias is not None:
        a = mB.bias.data.shape[0]
        mB.bias.data = mA.bias.data[:a]
'''
    def forward(self, input):
        # self._check_input_dim(input)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean[:input.shape[1]] = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean[:input.shape[1]]
                # update running_var with unbiased var
                self.running_var[:input.shape[1]] = exponential_average_factor * var * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_var[:input.shape[1]]
        else:
            mean = self.running_mean[:input.shape[1]]
            var = self.running_var[:input.shape[1]]
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :input.shape[1], None, None] + self.bias[None, :input.shape[1], None, None]
        return input
'''
def transfer_BN2d(mA, mB):
    if mA.weight is not None:
        mB.weight.data = mA.weight.data[:len(mB.weight.data)]
    if mA.bias is not None:
        mB.bias.data = mA.bias.data[:len(mB.bias.data)]
    if mA.running_mean is not None:
        mB.running_mean.data = mA.running_mean.data[:len(mB.running_mean.data)]
    if mA.running_var is not None:
        mB.running_var.data = mA.running_var.data[:len(mB.running_var.data)]
    if mA.num_batches_tracked is not None:
        mB.num_batches_tracked.data = mA.num_batches_tracked.data


config = {'con1': 8, 'con2': 4, 'con3': 4, 'con4': 4, 'con5': 4,
    'RCAB1': [8, 8], 'decon4': [4, 8], 'RCAB2': [8, 16], 'decon5': [4, 4],
    'RCAB3': [12, 16], 'decon6': [4, 4], 'RCAB4': [12, 16], 'decon7': [8, 8], 'decon8': [8, 8]}
model_Ori = Student_G(8).cuda()
model_Ori.configs = config
model_export = Expot(config).cuda()
model_Ori.load_state_dict(
    torch.load('/home/lizl/snap/third-stage/Super_NoDWT/Super_result_train/models/200-global_G.ckpt', map_location=lambda storage, loc: storage))  # 加载生成模型‘100-global_G.ckpt’
# model_export.load_state_dict(
#     torch.load('/home/lizl/snap/third-stage/Export/export-200-global_G.ckpt', map_location=lambda storage, loc: storage))  # 加载生成模型‘100-global_G.ckpt’

# tranfer_con
# input = torch.randn((1, 3, 256, 256)).cuda()
con_name = ['con1', 'con2', 'con3', 'con4', 'con5']
for name in con_name:
    ori = model_Ori.__getattr__(name).con
    exp = model_export.__getattr__(name)
    for mA, mB in zip(ori, exp):
        # pass
        if isinstance(mB, nn.Conv2d):
            transfer_ConV2d(mA, mB)
        if isinstance(mB, nn.BatchNorm2d):
            transfer_BN2d(mA, mB)
RCAB_name = ['RCAB1', 'RCAB2', 'RCAB3', 'RCAB4']
decon_name = ['decon4', 'decon5', 'decon6', 'decon7']
for r_name in RCAB_name:
    ori = model_Ori.__getattr__(r_name).body
    exp = model_export.__getattr__(r_name).body[0:5]
    for mA, mB in zip(ori, exp):
        if mA._get_name() == 'SuperSeparableConv2d':
            mA = mA.conv
            for m_A, m_B in zip(mA, mB):
                if isinstance(m_B, nn.Conv2d):
                    transfer_ConV2d(m_A, m_B)
                if isinstance(m_B, nn.BatchNorm2d):
                    transfer_BN2d(m_A, m_B)
        if mA._get_name() == 'SuperBatchNorm2d':
            transfer_BN2d(mA, mB)
    ori = model_Ori.__getattr__(r_name).CA.conv_du
    exp = model_export.__getattr__(r_name).body[5].conv_du
    for mA, mB in zip(ori, exp):
        if isinstance(mB, nn.Conv2d):
            transfer_ConV2d(mA, mB)
for d_name in decon_name:
    ori = model_Ori.__getattr__(d_name).decon
    exp = model_export.__getattr__(d_name)
    for mA, mB in zip(ori, exp):
        if isinstance(mB, nn.Conv2d):
            transfer_ConV2d(mA, mB)
        if isinstance(mB, nn.ConvTranspose2d):
            transfer_ConV2d(mA, mB)
        if isinstance(mB, nn.BatchNorm2d):
            transfer_BN2d(mA, mB)
d_name = 'decon8'
ori = model_Ori.__getattr__(d_name)
exp = model_export.__getattr__(d_name)
for mA, mB in zip(ori, exp):
    if isinstance(mB, nn.Conv2d):
        transfer_ConV2d(mA, mB)
    if isinstance(mB, nn.ConvTranspose2d):
        transfer_ConV2d(mA, mB)
    if isinstance(mB, nn.BatchNorm2d):
        transfer_BN2d(mA, mB)

input = torch.randn([1, 3, 256, 256]).cuda()
ori = model_Ori(input)
exp = model_export(input)
torch.allclose(exp, ori)
print('Max diff: ', (exp - ori).abs().max())
torch.save(model_export.state_dict(), './export-146-global_G.ckpt')
# print(ori==exp)
# print(1)

