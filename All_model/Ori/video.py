import torch
from torchvision.utils import make_grid
from nets.Teacher_Generator import Teacher_G
import argparse
import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils import data
import cv2
import queue
import threading
import time


def denorm( x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

config = {'con1': 8, 'con2': 4, 'con3': 4, 'con4': 4, 'con5': 4,
                'RCAB1': [8, 8], 'decon4': [4, 8], 'RCAB2': [8, 16], 'decon5': [4, 4],
                'RCAB3': [12, 16], 'decon6': [4, 4], 'RCAB4': [12, 16], 'decon7': [8, 8], 'decon8': [8, 8]}
student_G = Teacher_G(16).cuda()
student_G.load_state_dict(
            torch.load('./87-global_G.ckpt', map_location=lambda storage, loc: storage))
input = torch.randn((1, 3, 256, 256)).cuda()
student_G(input)
cap = cv2.VideoCapture('./21.mp4')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 352))
    print(ret)
    cv2.imshow('12', frame)
    frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.unsqueeze(0).cuda()
    img = denorm(student_G(frame)).cpu()
    # img_tensor = self.denorm(clean_fake1)  # 该步和batchsize有关
    grid = make_grid(img, nrow=1, padding=0)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imshow('11', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print('stop_dispose')
        break