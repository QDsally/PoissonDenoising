import numpy as np
import torch
import glob
import os
from skimage import io
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from model import DPDNN
from Dataset import Train_Data
# from newdataset import Train_Data1
from config import opt
from torch.utils.data import DataLoader
#from visdom import Visdom
from PIL import Image
from torchvision import transforms as T
from utils import PSNR
#from visvis import *
import pandas as pd
from torch.autograd import Variable
from utils import *
import cv2
from tensorboardX import SummaryWriter
def normalize(data):
    return data/255.

df = pd.DataFrame(columns=['epoch','psnr','loss'])
df.to_csv("G:\\LQ\\QQ\\colorDPDNNnew20.csv",index=False)
def train(use_gpu=True):

    train_data = Train_Data(opt.data_root)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)
    # val_data = Train_Data2(opt.data_root1,opt.data_root2)
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
    psnr_val = 0
    Val_Psnr_Max = 0
    Max = 0
    step1=1
    show_epoch = 0
    psnrs = []
    all_epoch = []
    epoches = 1
    loss_total = []
    loss_average = 0
    iteration = 1
    loss_average1 = 0
    for epoch in range(opt.max_epoch):

        # iteration = []
        if (epoch + 1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
        for i, (data, label) in enumerate(train_loader):
            data = data.cuda()
            label = label.cuda()
            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            # data = Variable(data.cuda())
            # label = Variable(label.cuda())
            output = net(data)
            loss = criterion(output, label)
            # iteration.append(i)
            # loss_total.append(loss)
            loss.backward()
            optimizer.step()
            num_batch += 1
            net.eval()
            # out_train = torch.clamp(data - net(data), 0., 1.)
            psnr_train = batch_PSNR(output, label, 1.)
            psnr_val += psnr_train
            loss_average += loss.item()
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(train_loader), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
    # writer.close()
            step += 1
            if iteration % 50 == 0:
                '''plt.figure()
                plt.plot(iteration,loss_total,label="loss")
                plt.draw()
                plt.show()'''
                loss_average1 = loss.item()
            # step1 += 1
            iteration += 1

        psnr_val /= 2000
        all_epoch.append(epoches)
        # psnrs.append(psnr_val)
        '''if epoches % 2 == 0:
            plt.figure()
            plt.plot(all_epoch, psnrs, label="loss")
            plt.draw()
            plt.show()
        epoches += 1'''
        tempt1 = False
        if Max < psnr_val:
            Max = psnr_val
            tempt1 = True
        if tempt1:
            torch.save(net.state_dict(), os.path.join(opt.outf, 'save7color.pth'))
        print

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        net.eval()
        psnr_val1 = 0
        loss_average2=0
        with torch.no_grad():
            psnr_test = 0
            for img_index in range(1, 33):
                if img_index < 10:
                    path = 'colorbsd32'
                    # index = 'test00'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name =  name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                elif img_index < 100:
                    path = 'colorbsd32'
                    index = 'test0'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name =  name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                elif img_index < 1000:
                    path = 'colorbsd32'
                    index = 'test'
                    name1 = str(img_index)
                    name2 = '.jpg'
                    name =  name1 + name2
                    files_source = glob.glob(os.path.join(path, name))
                for u in files_source:
                    # image
                    Img = io.imread(u)
                    Img1 = normalize(np.float32(Img[:, :, 0]))
                    Img1 = np.expand_dims(Img1, 0)
                    Img1 = torch.Tensor(Img1)
                    Img2 = normalize(np.float32(Img[:, :, 1]))
                    Img2 = np.expand_dims(Img2, 0)
                    Img2 = torch.Tensor(Img2)
                    Img3 = normalize(np.float32(Img[:, :, 2]))
                    Img3 = np.expand_dims(Img3, 0)
                    Img3 = torch.Tensor(Img3)
                    ISource = torch.cat((Img1, Img2, Img3), 0)
                    # Img = np.expand_dims(Img, 0)
                    ISource = np.expand_dims(ISource, 0)
                    ISource = torch.Tensor(ISource)
                    INoisy = add_noise(ISource)
                    ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
                with torch.no_grad():  # this can save much memory
                    Out = torch.clamp(net(INoisy), 0., 1.)
                loss1 = criterion(Out, ISource)
                    # loss.backward()
                loss_average1 = loss1.item()
                loss_average2 += loss_average1
                    # psnr = batch_PSNR(Out, ISource, 1.)
                psnr_val1 += batch_PSNR(Out, ISource, 1.)
            psnr_val1 /= 32
            psnrs.append(psnr_val1)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val1))
            writer.add_scalar('PSNR on validation data', psnr_val1, epoch)
            tempt = False
            if Val_Psnr_Max < psnr_val1:
                Val_Psnr_Max = psnr_val1
                tempt = True
            # log the images

        # epo.append(i)
        # loss_total.append(loss)
        # if (epoch+1) % 3 == 0:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lrdecay
        #     print('learning rate: ', optimizer.param_groups[0]['lr'])
        # save model
        loss_average2 /= 32
        list = [epoches,psnr_val1,loss_average2]
        data = pd.DataFrame([list])
        data.to_csv('G:\\LQ\\QQ\\colorDPDNNnew20.csv',mode='a',header=False,index=False)
        epoches += 1
        if tempt:  
            torch.save(net.state_dict(), os.path.join(opt.outf, 'colorDPDNNnew20.pth'))
        torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
    # torch.save(net.state_dict(), opt.save_model_path)
        # writer.close()


    print(psnrs)

    print('Finished Training')



        # learning rate decay




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





