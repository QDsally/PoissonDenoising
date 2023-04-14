
import torch
# from skimage.metrics import structural_similarity as ssim
import numpy as np
import math
from skimage.measure.simple_metrics import compare_psnr
# input type:tensor ; output type:tensor


# def add_testgaosinoise(input_img, noise_sigma):
#     noise_sigma = noise_sigma / 255

def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def add_testgaosinoise(input_img):
    noise_sigma = 0.1
    noise_img = torch.clamp(input_img+input_img*noise_sigma*torch.randn_like(input_img), 0.0, 1.0)

    return noise_img

def add_gaosi(img):
    # transform = T.ToTensor()
    # noise = torch.FloatTensor(img.size()).normal_(mean=0, std=25.5/255.)
    noise = torch.FloatTensor(img.size()).normal_(mean=1, std=0.1)
    # noise_img = (1.08)*noise*img
    noise_img = torch.clamp((1.08)*noise*img, 0.0, 1.0)

    # noise_img = torch.clamp( noise * img, 0.0, 1.0)
    #noise_img = noise_img.transpose(1,0,2)
    return noise_img

def add_rayleigh_noise(img):
    # transform = T.ToTensor()

    noise = np.random.rayleigh(0.1,size=img.size())
    # noise = np.random.gamma(shape=32, scale=40,size=img.size())

    #image_out = image + noise
    #image_out = transform(image_out)
    #image_out = normalize(image_out)*255.
    #image_out = normalize(image_out)

    #image_out = torch.clamp(image_out,0.0,1.0)
    noise = torch.FloatTensor(noise)
    noise = noise/255.
    #noise = torch.log(noise+1e-6)
    noise_img = torch.clamp(img + img*noise,0.0,1.0)
    #noise_img = noise_img.transpose(1,0,2)
    return noise_img



def add_noisegamma(img):
    # noise_sigma = noise_sigma / 255
    # noise_img = torch.clamp(input_img+noise_sigma*torch.randn_like(input_img), 0.0, 1.0)
    noise = np.random.gamma(shape=1, scale=1, size=img.size())

    # image_out = image + noise
    # image_out = transform(image_out)
    # image_out = normalize(image_out)*255.
    # image_out = normalize(image_out)
    # noise = noise / 255.
    # print(noise)
    # image_out = torch.clamp(image_out,0.0,1.0)
    noise = torch.FloatTensor(noise)
    # noise = noise / 255.
    # noise = torch.log(noise+1e-6)
    noise_img = torch.clamp((0.2)*img*noise+img, 0.0, 1.0)
    # print(noise_img)
    # noise_img = noise_img.transpose(1,0,2)
    return noise_img
    # return noise_img

def add_noise(img):
    # noise=add_rayleigh_noise(img,0.8,0.1)
    img = img.data.cpu().numpy().astype(np.float32)
    noise = np.random.poisson(img * 2)/2
    # print(noise)
    noise = torch.FloatTensor(noise)
    noise = torch.clamp(noise, 0.0, 1.0)
    #print(noise)
    return noise

# def add_rayleigh_noise(img,a,b):
    # return a+(((-b)*(torch.log(1-torch.rand_like(img))))**(0.5))

def PSNR(img1, img2, color=False):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse * 255 * 255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

add_noisegamma(torch.rand(1,128,128))


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])














