import torch
from PIL import Image
import time
import torchvision
import numpy as np
from torchvision import transforms as T
from utils import add_noise
from utils import PSNR
import math
# from newmodel import DPDNN
# from HQSmodel import DPDNN
from model3 import DPDNN
import torch.nn as nn
from config import opt
from utils import *
import os
import glob
# Here is the path of your test image, 'i' means the ith image, you only need to provide the ground truth image
# Then we add Gaussian noise to the gt image
i = 0
label_img = './Set68/test066.png'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform1 = T.ToTensor()
transform2 = T.ToPILImage()
allTime = 0 
averageTime = 0
with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # net.load_state_dict(torch.load(os.path.join('logs', 'chongxin1.pth')))
    net.load_state_dict(torch.load(os.path.join('logs', 'NewtonPeak=14.pth')))
    # net.load_state_dict(torch.load(os.path.join('logs', 'newTrain1.pth')))
    files_source = glob.glob(os.path.join('Set12', '*.png'))
    psnr_test = 0
    i = 0
    for f in files_source:
        img = Image.open(f)
    # img.show()
        #pad = torchvision.transforms.Pad((0, 0, 191, 31), padding_mode='symmetric')
        #img = pad(img)
        # print(img)
        label = np.array(img).astype(np.float32)  # label:0~255
        img_H = img.size[0]
        img_W = img.size[1]
        img = transform1(img)
        # print(img.shape)
        img_noise = add_noise(img).resize_(1, 1, img_H, img_W)
        s1 = time.time()
        output = net(img_noise)
        s2 = time.time()
        # psnr_test = batch_PSNR(output,img,1.)
        output = output
        img_noise = img_noise.resize_(img_H, img_W)
        img_noise = torch.clamp(img_noise, min=0, max=1)
        # img_noise = transform2(img_noise)

    # To save the output(denoised) image, you must create a new folder. Here is my path.
    #     output.save('./proposed/newSet128/12.png')

        img_noise = transform2(img_noise.resize_(img_H, img_W))
    # img_noise.show()
        i = i + 1
        img_noise.save('./HQSGRAY/BSD531/noise.png')
        
    # show output image
    # output.show()
        output = np.array(output)  # output:0~255

    # Because of the randomness of Gaussian noise, the output results are different each time.
    # print(i, 'MSE loss:%f, PSNR:%f'%(batch_PSNR(output, label,1.)))

        print('MSE loss:%f, PSNR:%f' % (PSNR(output, label, 1.)))
        print('Elapsed Time: %0.3f s' % (s2 - s1))
        allTime = allTime + (s2 - s1)
        psnr_test += PSNR(output, label, 1.)[1]
    psnr_test /= len(files_source)
    averageTime = allTime / len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('Elapsed AverageTime: %0.3f s' % averageTime)










