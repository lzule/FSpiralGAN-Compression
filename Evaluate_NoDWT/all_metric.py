# -*-coding:utf-8-*-
import sys
from tqdm import tqdm
import torch
from torchvision.utils import save_image
import os
from PIL import Image
# import importlib
# from Stu import Decoder
from Student_Generator import Student_G as Decoder
# import argparse
import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils import data
import numpy as np
# import time
import click
import click
import os
# import numpy as np
import cv2
import math
# from tqdm import tqdm
from skimage import io, color, filters
from data_loader import TrainDataLoader
from torchvision.utils import make_grid
from thop import profile
from thop import clever_format
import random


def to_numpy(clean, muddy):
        clean_ = denorm(clean.data.cpu())
        muddy_ = denorm(muddy.data.cpu())
        # img_tensor = self.denorm(clean_fake1)   # 该步和batchsize有关
        clean_grid = make_grid(clean_, nrow=1, padding=0)
        clean_ndarr = clean_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        muddy_grid = make_grid(muddy_, nrow=1, padding=0)
        muddy_ndarr = muddy_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return clean_ndarr, muddy_ndarr

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = np.int32(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int32(al1 * len(rgl))
    T2 = np.int32(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)
            if not bottom:
                continue
            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)


def psnr(target,prediction):
    PIXEL_MAX=255.0
    target = target.astype(np.float64)
    prediction = prediction.astype(np.float64)
    mse=np.mean((target - prediction) ** 2 )
    psnr_val = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr_val


def ssim(img1, img2):
    C1 = (0.01 * 255.0)**2
    C2 = (0.03 * 255.0)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


transform = list()
transform.append(T.Resize([704, 1280]))  # 注意要为256的倍数
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

def get_model(model_save_dir, epoch , global_G_ngf, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = Decoder(global_G_ngf).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_dir, str(epoch) + '-global_G.ckpt'), map_location=lambda storage, loc: storage))
    return model

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def all_configs(n_channels):
    def yield_channels(i):
        if i == len(n_channels):
            yield []
            return
        for n in n_channels[i]:
            for after_channels in yield_channels(i + 1):
                yield [n] + after_channels
    for i, channels in enumerate(yield_channels(0)):
        yield {'channels': channels}

def Sample(n_channels):
    # configs = all_configs(n_channels)
    weights = None
    configs = dict()
    for key,value in n_channels.items():
        if key[:3] == 'con':
            configs[key] = random.choices(value, weights=weights)[0]
        else:
            temp = list()
            temp.append(random.choices(value[0], weights=weights)[0])
            temp.append(random.choices(value[1], weights=weights)[0])
            configs[key] = temp
    return configs
    

@click.command()
@click.option('--name',
            type=click.STRING,
            default='distill',
            help='Path to the folder containing the ground-truth images')

@click.option('--ngf',
            type=click.INT,
            default=8,
            help='Path to the folder containing the ground-truth images')

# @click.option('--image_dir',
#             type=click.STRING,
#             default='/home/lizl/snap/second-stage/evaluate/SUID/Synthetic_Underwater_images',
#             help='Path to the folder containing the ground-truth images')

@click.option('--gpu_id',
            type=click.INT,
            default=0,
            help='Path to the folder containing the ground-truth images')

# use: python all_metric.py --


def test(name, ngf, gpu_id, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Load the trained generator.
    # print(ngf)
    channels_log = list()
    n_channels = {'con1':[8, 4], 'con2':[8, 4], 'con3':[8, 4], 
            'con4':[8, 4],'con5':[8, 4],
            'RCAB1':[[8, 4], [8, 4]], 'decon4':[[8, 4], [8, 4]],
            'RCAB2':[[16, 12, 8, 4], [16, 12, 8, 4]], 'decon5':[[8, 4], [8, 4]],
            'RCAB3':[[16, 12, 8, 4], [16, 12, 8, 4]], 'decon6':[[8, 4], [8, 4]],
            'RCAB4':[[16, 12, 8, 4], [16, 12, 8, 4]], 'decon7':[[8, 4], [8, 4]],
            'decon8':[[8, 4], [8, 4]]}
    
    model_save_dir = '/home/lizl/snap/third-stage/' + name + '/Super_result_train/models'
    print_log = open(os.path.join('/home/lizl/snap/third-stage/evaluate-NoDWT', name + 'metirc.txt'), 'w')
    print(print_log)
    torch.cuda.set_device(gpu_id)
    begin_epoch = 200
    end_epoch = 201
    step = 1
    for k in range(begin_epoch, end_epoch, step):
        print('epoch:[{}]:--------------------------starting--------------------------'.format(k))
        generator = get_model(model_save_dir, epoch=k, global_G_ngf=ngf) 
        dataloader = TrainDataLoader('/home/lizl/snap/second-stage/data/val', 1, 256,
                                   256, 4)()
        for _ in tqdm(range(5000)):
            generator.configs = Sample(n_channels)
            psnr_score = []
            ssim_score = []
            uiqm_score = []
            uciqe_score = []
            with torch.no_grad():
                for i, tensor_dic in enumerate(tqdm(dataloader)):
                    muddy, clean = tensor_dic["muddy"], tensor_dic["clean"].to(device)
                    muddy_ = muddy.to(device)
                    clean_fake = generator(muddy_)
                    # save_image((denorm(clean_fake.data.cpu())),
                    #            os.path.join('aaa' + '_.jpg'), nrow=2, padding=0)
                    gt_mg, fk_mg = to_numpy(clean, clean_fake)
                    uiqm,uciqe = nmetrics(fk_mg)
                    psnr_score.append(psnr(gt_mg,fk_mg))
                    ssim_score.append(ssim(gt_mg,fk_mg))
                    uiqm_score.append(uiqm)
                    uciqe_score.append(uciqe)
                    if i == 100:
                        break
            uiqm_va = np.mean(np.array(uiqm_score))
            uciqe_va = np.mean(np.array(uciqe_score))
            psnr_va = np.mean(np.array(psnr_score))
            ssim_va = np.mean(np.array(ssim_score))
            # 计算参数量和FLOPS
            flops, params = profile(generator, inputs=(muddy_, ))
            flops, params = clever_format([flops, params], "%.3f")
            print('{}:epoch[{}]:---flops: {}---params: {}---Psnr_score: {}---Ssim_score:{}---uiqm_va: {}---uciqe_va:{}'.format(generator.configs, k, flops, params, psnr_va, ssim_va, uiqm_va, uciqe_va))
            print('{}:epoch[{}]:---flops: {}---params: {}---Psnr_score: {}---Ssim_score:{}---uiqm_va: {}---uciqe_va:{}'.format(generator.configs, k, flops, params, psnr_va, ssim_va, uiqm_va, uciqe_va), file=print_log)


if __name__ == '__main__':
    # model_save_dir = '/home/lizl/snap/second-stage/distill/result_train/models'
    # ngf = 8
    # image_dir = '/home/lizl/snap/second-stage/evaluate/SUID/Synthetic_Underwater_images'
    # result_dir = os.path.join('/home/lizl/snap/second-stage/evaluate', 'distill')
    test()
