import numpy as np
import torch
import torch.nn as nn
from torch import optim
# from model import DPDNN
#from zhuyilimodel import DPDNN
from newmodel import DPDNN
from Dataset import Train_Data
from config import opt
from SSIM import *
from torch.utils.data import DataLoader
#from visdom import Visdom
from PIL import Image
from torchvision import transforms as T
from utils import PSNR
import glob
import pandas as pd
import cv2
import os
from torch.autograd import Variable
#from visvis import *
from utils import *
from tensorboardX import SummaryWriter

def normalize(data):
    return data/255.

# dfLOSS = pd.DataFrame(columns=['iteration','loss'])
# dfLOSS.to_csv("chongxingaide\\ADMM.csv",index=False)
dfPSNR = pd.DataFrame(columns=['epoch','psnr','ssim','lossval','losstrain'])
dfPSNR.to_csv("chongxingaide\\ADMMPeak=1.csv",index=False)
transform2 = T.ToPILImage()
def train(use_gpu=True):

    train_data = Train_Data(opt.data_root1)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = DPDNN()
    criterion = nn.MSELoss()
    if use_gpu:
        net = net.cuda()
        net = nn.DataParallel(net)
        criterion = criterion.cuda()

    # initialize weights by Xavizer
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    # Save the original model


    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    num_batch = 0
    num_show = 0
    writer = SummaryWriter(opt.outf)
    # visdom
    # vis = Visdom()
    step = 0
    loss_average = 0
    iteration = 1
    epoches = 1
    Val_Psnr_Max = 0
    listPSNR=[]
    listSSIM=[]
    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()
            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            output = net(data)
            # loss = criterion(output, label)
            loss = criterion(output, label)
            loss.backward()
            loss_average += loss.item()
            optimizer.step()
            num_batch += 1
            net.eval()
            # out_train = torch.clamp(data - net(data), 0., 1.)
            psnr_train = batch_PSNR(output, label, 1.)
            # label = label.resize_(256, 256)
            # output = output.resize_(256, 256)
            # output = transform2(output)
            # label = transform2(label)
            # ISource = np.array(label)
            # Out = np.array(output)
            # ssim = compute_ssim(Out, ISource)

            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f " %
                  (epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            if iteration % 500 == 0:
                '''plt.figure()
                plt.plot(iteration,loss_total,label="loss")
                plt.draw()
                plt.show()'''
                loss_average1 = loss.item()
                # print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f " %
                #       (epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))
            # step1 += 1
            iteration += 1
        net.eval()
        psnr_val1 = 0
        ssim_val1 = 0
        # listPSNR=[]
        # listSSIM=[]
        loss_average2=0
        with torch.no_grad():
            psnr_test = 0
            for img_index in range(1, 51):
                if img_index < 10:
                    path = 'yanzhengji'
                    # index = 'test00'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name = name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                elif img_index < 100:
                    path = 'yanzhengji'
                    # index = 'test0'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name = name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                elif img_index < 1000:
                    path = 'yanzhengji'
                    # index = 'test'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name = name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                for u in files_source:
                    # image
                    Img = cv2.imread(u)
                    Img = normalize(np.float32(Img[:, :, 0]))
                    Img = np.expand_dims(Img, 0)
                    Img = np.expand_dims(Img, 1)
                    ISource = torch.Tensor(Img)
                    INoisy = add_noise(ISource)
                    ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
                with torch.no_grad():  # this can save much memory
                    Out = torch.clamp(net(INoisy), 0., 1.).cuda()
                loss1 = criterion(Out, ISource)
                    # loss.backward()
                loss_average1 = loss1.item()
                    # psnr = batch_PSNR(Out, ISource, 1.)
                psnr_val1 += batch_PSNR(Out, ISource, 1.)
                # loss_average1 = loss.item()
                loss_average2 += loss_average1
                ISource = ISource.resize_(256, 256)
                Out = Out.resize_(256, 256)
                ISource = transform2(ISource)
                Out = transform2(Out)
                ISource = np.array(ISource)
                Out = np.array(Out)
                ssim_val1 += compute_ssim(Out, ISource)
            psnr_val1 /= 50
            ssim_val1 /= 50
            listPSNR.append(psnr_val1)
            listSSIM.append(ssim_val1)
            # psnrs.append(psnr_val1)
            print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch + 1, psnr_val1,ssim_val1))
            writer.add_scalar('PSNR on validation data', psnr_val1, epoch)
            tempt = False
            if Val_Psnr_Max < psnr_val1:
                Val_Psnr_Max = psnr_val1
                tempt = True
        loss_average /= 4000

        # if iteration % 500 == 0:
        #     '''plt.figure()
        #     plt.plot(iteration,loss_total,label="loss")
        #     plt.draw()
        #     plt.show()'''
        loss_average2 /= 50
            # print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f " %
            #       (epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))
        # step1 += 1
        # iteration += 1
        # list1 = [iteration, loss_average1]
        # data1 = pd.DataFrame([list1])
        # data1.to_csv('C:\\Users\\pan\\Desktop\\LQ\\QQ\\trainLOSSiteration66.csv', mode='a', header=False, index=False)
        list2 = [epoches,psnr_val1,ssim_val1,loss_average2,loss_average]
        data2 = pd.DataFrame([list2])
        data2.to_csv('chongxingaide\\ADMMPeak=1.csv', mode='a', header=False, index=False)
        epoches += 1
        if tempt:
            # torch.save(net.state_dict(), os.path.join(opt.outf, 'save7noisefinal4.pth'))
            torch.save(net.state_dict(), os.path.join(opt.outf, 'bijiao1.pth'))
        torch.cuda.empty_cache()

        # learning rate decay
        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
        torch.save(net.state_dict(), opt.save_model_path)
    print('Finished Training')
    # print("\nPSNR_val: %.4f SSIM_val: %.4f" % ( listPSNR, ssim_val1))
    print(listPSNR)
    print(listSSIM)

# This function is for checking the training effect, not the test code
'''def test(epoch, i):
    net1 = DPDNN()
    net1 = net1.cuda()
    net1 = nn.DataParallel(net1)
    net1.load_state_dict(torch.load(opt.load_model_path))

    noise = Image.open('./data/test_data/sigma%d/test.png'%opt.noise_level)
    label = Image.open('./data/test_data/sigma%d/testgt.png'%opt.noise_level)

    img_H = noise.size[0]
    img_W = noise.size[1]

    transform = T.ToTensor()
    transform1 = T.ToPILImage()
    noise = transform(noise)
    noise = noise.resize_(1, 1, img_H, img_W)
    noise = noise.cuda()
    label = np.array(label).astype(np.float32)

    output = net1(noise)     # dim=4
    output = torch.clamp(output, min=0.0, max=1.0)
    output = torch.tensor(output)
    output = output.resize(img_H, img_W).cpu()
    output_img = transform1(output)

    # every 500 batches save test output
    if i%500 == 0:
        save_index = str(int(epoch*(opt.num_data/opt.batch_size/500) + (i+1)/500))
        output_img.save('./data/test_data/sigma%d/test_output/'%opt.noise_level+save_index+'.png')

    output = np.array(output_img)
    mse, psnr = PSNR(output, label)
    return mse, psnr'''


if __name__ == '__main__':
    train()





