import cv2
import os
import argparse
import glob
from SSIM import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from newmodel import DPDNN
from utils import *
import PIL
import torchvision
from skimage import io
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
transform2 = T.ToPILImage()
'''def to_numpy_image(tensor):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
        if array.shape[2] == 1:
            array = array[:, :, 0]
    # Clip image to [0, 1]
    array[array < 0] = 0
    array[array > 1] = 1
    return array'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='xiaoshuju', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DPDNN()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # device = torch.device('cpu')
    model.load_state_dict(torch.load(os.path.join('logs','chongxincolor1.pth')))

    #卷积核可视化
    # plt.rcParams['figure.dpi'] = 300
    # for layer in model.modules():
    #     # print(layer)
    #     name=layer.__class__.__name__
    #     # print(name)
    #     if name=="TNRDlayer":
    #         # print(layer)
    #         K=layer.weight.cuda()#为什么他是可改变的
    #         # K=net.layers[1].get_weights()
    #         # print(K)
    #         MFilters=getGaborFilterBank(8,64,7,7).cuda()
    #         # print(MFilters)
    #         ku = K.multiply(MFilters)
    #         Ku=ku.detach().cpu().numpy()
    #         # print(ku)
    #         # print(Ku)
    #         n_filters=64
    #         ix=1
    #         for i in range(n_filters):
    #             ax = plt.subplot(8, 8, ix)
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             plt.imshow(Ku[i, :, :], cmap='gray')
    #             plt.axis('off')
    #             # plt.imshow(MFilters[i, :, :], cmap='gray')
    #             ix+=1
    #         plt.show()
    #         break
    #径向基可视化
    # num=0
    # plt.rcParams['figure.dpi'] = 100
    # for layer in net.modules():
    #     # print(layer)
    #     name=layer.__class__.__name__
    #     # print(name)
    #     if name=="RBF":
    #         # print(layer)
    #         num=num+1
    #         print(num)
    #         K=layer.weight#为什么他是可改变的
    #         K=K[:,:,4]
    #         K = K.view(-1)
    #         # K=K.detach().numpy()
    #         print(K)
    #         K = K.detach().cpu().numpy()
    #         # print(K)
    #         # print(K.size())
    #         C=layer.centers
    #         # print(C)
    #         C=C.detach().cpu().numpy()
    #         # print(C.size())
    #         # print(C)
    #         # plt.plot(C,K)
    #         # plt.show()
    #         # break

    # model.eval()
    # load data info
    # print('Loading data info ...\n')
    # files_source = glob.glob(os.path.join('data','l.bmp'))
    # files_source = glob.glob(os.path.join('xiaoshuju', 'imagenet_2012_000001.png'))
    # files_source = glob.glob(os.path.join('xiaoshuju', 'imagenet_2012_000001.png'))
    files_source = glob.glob(os.path.join('TEST2', '*.bmp'))
    files_source.sort()
    # process data
    psnr_test = 0
    i=100
    for f in files_source:
        # image
        i=i+1
        # Img = cv2.imread(f)
        Img = io.imread(f)
        Img1 = normalize(np.float32(Img[:,:,0]))
        Img1 = np.expand_dims(Img1, 0)
        Img1 = torch.Tensor(Img1)    
        Img2 = normalize(np.float32(Img[:, :, 1]))
        Img2 = np.expand_dims(Img2, 0)
        Img2 = torch.Tensor(Img2)
        Img3 = normalize(np.float32(Img[:, :, 2]))
        Img3 = np.expand_dims(Img3, 0)
        Img3 = torch.Tensor(Img3)
        Img=torch.cat((Img1,Img2,Img3),0)
        # Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 0)
        ISource = torch.Tensor(Img)
        # noise
        # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        # INoisy = ISource + noise
        # IINoisy=transform2(INoisy)
        # IINoisy.show()
        INoisy = add_noise(ISource)
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            out=model(INoisy)
        Out = torch.clamp(out, 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        # ISource = ISource.resize_(256, 128)
        # Out = Out.resize_(512, 512)
        # Out = Out.resize_(256, 256)
        # INoisy = INoisy.resize_(180, 180)
        # ISource = transform2(ISource)
        # Out = transform2(Out)
        # INoisy = transform2(INoisy)
        # ISource = np.array(ISource)
        # Out = np.array(Out)
        # INoisy = np.array(INoisy)
        # ssim_val = compute_ssim(Out, ISource)
        # INoisy = INoisy.resize_(224, 224)
        # INoisy = transform2(INoisy)
        # INoisy.show()
        psnr_test += psnr
        # print("PSNR %f" % ( psnr))
        print("%s PSNR %f" % (f, psnr))
        # print("\nSSIM_val: %.4f" % ssim_val)
        # path1=os.path.join('data', 'result')
        torchvision.utils.save_image(Out, './2022/CBSD531/%d.bmp'%i)
        # torchvision.utils.save_image(Out, './CBSD1520/%d.bmp' % i)
        # io.imsave('\data\%d.png'%i,Out)
        # OOut = transform2(Out)
        # Out.save('./TRDPDSet121/%d.bmp'%i)
        # i+=1
        # Out.show()
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

'''def getGaborFilterBank(nScale, M, h, w):
    Kmax = math.pi / 2
    f = math.sqrt(2)
    sigma = math.pi
    sqsigma = sigma ** 2
    postmean = math.exp(-sqsigma / 2)

    if h != 1:
        MM = int(M / nScale)
        gfilter_real = torch.zeros(nScale, MM, h, w)
        # Gfilter_real=torch.zeros(M*nScale, h, w)

        for j in range(nScale):

            for i in range(MM):
                theta = i / MM * math.pi
                # k = Kmax / f ** (nScale - 1)
                k = Kmax / f ** (j + 1)
                xymax = -1e309
                xymin = 1e309
                for y in range(h):
                    for x in range(w):
                        # A = np.zeros([h, w])
                        # A[x, y] = 1

                        y1 = y + 1 - ((h + 1) / 2)
                        x1 = x + 1 - ((w + 1) / 2)
                        tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
                        tmp2 = math.cos(k * math.cos(theta) * x1 + k * math.sin(theta) * y1) - postmean  # For real part
                        # tmp3 = math.sin(k*math.cos(theta)*x1+k*math.sin(theta)*y1) # For imaginary part
                        gfilter_real[j][i][y][x] = k * k * tmp1 * tmp2 / sqsigma
                        xymax = max(xymax, gfilter_real[j][i][y][x])
                        xymin = min(xymin, gfilter_real[j][i][y][x])
                gfilter_real[j][i] = (gfilter_real[j][i] - xymin) / (xymax - xymin)
        gfilter_real = gfilter_real.view(1, -1, h, w)
        gfilter_real = gfilter_real.squeeze(0)

        #     G[j]=gfilter_real[i]
        # Gfilter_real=torch.cat((gfilter_real[j][i]),1)
    else:
        gfilter_real = torch.ones(M, h, w)
    # gfilter_real=gfilter_real.unsqueeze(0).repeat(M,1,1,1)

    return gfilter_real'''

if __name__ == "__main__":
    main()
