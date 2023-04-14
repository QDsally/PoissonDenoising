
import torch
from PIL import Image
import torchvision
import numpy as np
from torchvision import transforms as T
from utils import add_noise
from utils import PSNR
import math
from model import DPDNN
import torch.nn as nn
from config import opt
from utils import *
import os
# Here is the path of your test image, 'i' means the ith image, you only need to provide the ground truth image
# Then we add Gaussian noise to the gt image
i = 10
label_img = './Set12/01.png'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform1 = T.ToTensor()
transform2 = T.ToPILImage()

with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    net.load_state_dict(torch.load(opt.load_model_path))

    img = Image.open(label_img)
    # img.show()
    # pad = torchvision.transforms.Pad((0,0,191,31),padding_mode='symmetric')
    # img = pad(img)
    # print(img)
    # label = np.array(img).astype(np.float32)   # label:0~255
    # img_H = img.size[0]
    # img_W = img.size[1]
    # img = transform1(img)
    # print(img.shape)
    # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
    img_noise = add_noise(img, noise_sigma=15).resize_(1, 1, img_H, img_W)
    noise = img_noise - img
    # output = net(img_noise)
    # output = output
    noise = noise.resize_(img_H, img_W)
    noise = torch.clamp(noise, min=0, max=1)
    noise = transform2(noise)

    # To save the output(denoised) image, you must create a new folder. Here is my path.
    # output.save('./output/sigma%d/%d.png'%(opt.noise_level, i))

    # img_noise = transform2(img_noise.resize_(img_H, img_W))
    #img_noise.show()
    noise.save('./output/noise.png')

    # show output image
    # noise.show()
    # output = np.array(output)   # output:0~255

    # Because of the randomness of Gaussian noise, the output results are different each time.
    # print(i, 'MSE loss:%f, PSNR:%f'%(batch_PSNR(output, label,1.)))
    # print(i, 'MSE loss:%f, PSNR:%f' % (PSNR(output, label, 1.)))










