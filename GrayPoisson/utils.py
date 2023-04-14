
import torch
import numpy as np
import math
import utils
from skimage.measure.simple_metrics import compare_psnr
# from skimage.measure.simple_metrics import com
from skimage.measure import  compare_psnr,compare_ssim
# from skimage.metrics import structural_similarity as compare_ssim
# input type:tensor ; output type:tensor
def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

# def add_noise(input_img, noise_sigma):
#     noise_sigma = noise_sigma / 255
#     noise_img = torch.clamp(input_img+noise_sigma*torch.randn_like(input_img), 0.0, 1.0)

#     return noise_img

'''def add_poiss_noise_image(img):
  sy,sx = img.shape
  lambda_flat = np.reshape(img,[-1,1]).astype(np.float32)
  noisy_flat = np.random.poisson(lam=lambda_flat)
  noisy = np.reshape(noisy_flat,[sy,sx])
  return(noisy.astype(np.float32))'''

'''def add_noise(image):
    # image = image.data.cpu().numpy().astype(np.float32)
    gt = image
    gt = gt[0,:,:]
    # print(gt)
    max_val = np.amax(np.amax(gt))
    gt = gt.astype(np.float32) * (1.0 / float(max_val)) - 0.5
    img_peak = (0.5 + gt) * float(8)
    noisy = utils.add_poiss_noise_image(img_peak).astype(np.float32)
    noisy = (noisy / float(8)) - 0.5
    noise = torch.FloatTensor(noisy)
    noisy = torch.clamp(noise, 0.0, 1.0)
    # print(noisy.shape)
    return noisy'''
def add_noise(img):
    img = img.data.cpu().numpy().astype(np.float32)
    noise = np.random.poisson(img * 1)/1
    
    noise = torch.FloatTensor(noise)
    noise = torch.clamp(noise, 0.0, 1.0)
    
    return noise
def psnr4(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def add_rayleigh_noise(img,a,b):
    # return a+(((-b)*(torch.log(1-torch.rand_like(img))))**(0.5))

def PSNR(img1, img2, color=False):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse * 255 * 255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

add_noise(torch.rand(1,128,128))

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_PSNR1(img, imclean, data_range):
    # Img = img.data.cpu().numpy().astype(np.float32)
    # Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += compare_psnr(imclean[i,:,:,:], img[i,:,:,:])
    return (PSNR/img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])

# ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)











