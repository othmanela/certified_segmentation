# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import random
import logging
from datetime import datetime
from collections import OrderedDict

#the utils of scu
from utils import utils_logger
from utils import utils_modelscu
from utils import utils_model
from utils import utils_image as util

import torch
from torch.nn import functional as F
from torch.utils import data

import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
#from PIL import Image
import argparse

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def diffusion_args():
    defaults = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule='linear',
        timestep_respacing='',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False
    )
    return defaults

def scu_args():
    defaults = dict(
        model_name='scunet_color_25', #'scunet_color_25', # add 'scunet_color_50'
        testset_name='kodak24',
        noise_level_img=25, #25, #add 50
        x8=False,
        show_img=False,
        model_zoo='../lib/model_zoo/',
        testsets='testsets',
        results='results',
        need_degradation=False
    )
    return defaults

def dncnn_args():
    defaults = dict(
        model_name='dncnn_color_blind', #'dncnn_15, dncnn_25, dncnn_50, dncnn_gray_blind, dncnn_color_blind, dncnn3'
        testset_name='set12',
        noise_level_img=50,
        x8=False,
        show_img=False,
        model_pool='../lib/model_zoo/',
        testsets='testsets',
        results='results',
        need_degradation=False,
        task_current='dn',
        sf=1
    )
    return defaults

class BaseDataset(data.Dataset):
    def __init__(self, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 normalize=True):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate
        self.normalize = normalize

        self.files = []

    def __len__(self):
        return len(self.files)
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        if self.normalize:
            image = image / 255.0
            image -= self.mean
            image /= self.std
        return image
    
    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label
    
    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        return image, label

    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def gen_sample(self, image, label, 
            multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                    rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, 
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)
        
        image = image.transpose((2, 0, 1))
        
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def inference(self, model, image, flip=False, unscaled=False):
        if unscaled: assert not flip
        size = image.size()
        pred = model(image)
        if not unscaled:
            pred = F.upsample(input=pred, 
                                size=(size[-2], size[-1]), 
                                mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        device = torch.device("cuda:%d" % model.device_ids[0])
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)
        padvalue = -1.0  * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width, 
                                    self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width, 
                                        self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).to(device)
                count = torch.zeros([1,1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img, 
                                                      h1-h0, 
                                                      w1-w0, 
                                                      self.crop_size, 
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)

                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred


    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


    def multi_scale_inference_noisybatch(self, model, n, sigma, image_np, normalize=True, scales=[1], flip=False, unscaled=False):
        if unscaled: assert len(scales) == 1 and scales[0] <= 1.0
        ori_height, ori_width, _ = image_np.shape
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([n, self.num_classes,
                                    ori_height,ori_width]).cuda()
        images = []
        for i in range(n):
            img = image_np + sigma * np.random.randn(*image_np.shape) 
            if normalize:
               img -= self.mean
               img /= self.std
               images.append(img)
        for scale in scales:
            new_images = [self.multi_scale_aug(image=img,
                                           rand_scale=scale,
                                           rand_crop=False).astype(np.float32) for im in images]
            height, width = new_images[0].shape[:-1]
            new_images_s = np.stack(new_images).transpose((0, 3, 1, 2))
            if scale <= 1.0:
                preds = self.inference(model, torch.from_numpy(new_images_s), flip, unscaled=unscaled)
                if not unscaled:
                    preds = preds[:, :, 0:height, 0:width]
            else:
                rows = np.int(np.ceil(1.0 * (height - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (width - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([n, self.num_classes,
                                           height,width]).cuda()
                count = torch.zeros([n,1, height, width]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], height)
                        w1 = min(w0 + self.crop_size[1], width)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = torch.from_numpy(new_images_s[:, :, h0:h1, w0:w1])
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            if not unscaled:
                preds = F.upsample(preds, (ori_height, ori_width), 
                                    mode='bilinear')
                final_pred += preds
            else:
                final_pred = preds
        return final_pred

    def multi_scale_inference_denoisedbatch(self, model, n, sigma, image_np, dd_model, diffusion, normalize=False,
                                       scales=[1], flip=False, unscaled=False):
        if unscaled: assert len(scales) == 1 and scales[0] <= 1.0
        ori_height, ori_width, _ = image_np.shape
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([n, self.num_classes,
                                  ori_height, ori_width]).cuda()
        images = []

        for i in range(n):
            img_np = image_np + sigma * np.random.randn(*image_np.shape)
            ######
            args = dncnn_args()
            n_channels = 3  # fixed, 1 for grayscale image, 3 for color image
            if args['model_name'] in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
                nb = 20               # fixed
            else:
                nb = 17               # fixed

            model_path = os.path.join(args['model_pool'], args['model_name'] + '.pth')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            from models.network_dncnn import DnCNN as net
            modeldncnn = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
            modeldncnn.load_state_dict(torch.load(model_path), strict=True)
            modeldncnn.eval()
            for k, v in modeldncnn.named_parameters():
                v.requires_grad = False
            modeldncnn = modeldncnn.to(device)

            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []

            img_L = img_np
            img_L = util.single2tensor4(img_L)
            img_L = img_L.to(device)

            if not args['x8']:
                img_E = modeldncnn(img_L)
            else:
                img_E = utils_model.test_mode(modeldncnn, img_L, mode=3)

            img_out = img_E.data.squeeze().float().clamp_(0, 1)
            normalize = transforms.Compose([
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            img_out_norm = normalize(img_out)
            images.append(img_out_norm)

        for scale in scales:
            images_stacked = torch.stack(images)
            preds = self.inference(model, images_stacked, flip, unscaled=unscaled)
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        print(stride_h, stride_w, final_pred.shape)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()
                print(rows, cols)
                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred


    
    def convert_label(self, label, inverse=False):
        return label
