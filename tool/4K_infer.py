import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

import tqdm

class NAFNetDeblurer(nn.Module):
    def __init__(self, output_path, weight_path):
        super(NAFNetDeblurer, self).__init__()
        self.output_path = output_path
        self.weight_path = weight_path
        self.generator = torch.load(weight_path)
        self.generator.eval()

    def imread_uint(self, path, n_channels=3):
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.array(img)
        img = img.astype(np.uint8)
        return img

    def single2tensor3(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(img)
        return img.unsqueeze(0)

    def crop_predict(self, img_L):
        h, w, _ = img_L.shape
        sf = 2
        tile = 512
        overlap = 128
        h_idx_list = list(range(0, h - tile, tile - overlap))
        if h - tile - h_idx_list[-1] < overlap:
            h_idx_list = h_idx_list[:-1]
        w_idx_list = list(range(0, w - tile, tile - overlap))
        if w - tile - w_idx_list[-1] < overlap:
            w_idx_list = w_idx_list[:-1]

        E = torch.zeros_like(img_L)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                h_idx = int(h_idx)
                w_idx = int(w_idx)
                in_patch = img_L[:, :, h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.generator(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
                W[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask

        output = E.divide(W)
        return output

    def run_patches(self, images_path=None):
        image_files = self.get_images(images_path)
        for image_file in tqdm(image_files):
            img_L = self.imread_uint(image_file, 3)

            image_name = os.path.basename(image_file)
            img = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.output_path, image_name), img)

            tmps = image_name.split('.')
            assert len(
                tmps) == 2, f'Invalid image name: {image_name}, too much "."'
            restoration_save_path = os.path.join(
                self.output_path, f'{tmps[0]}_restoration.{tmps[1]}')

            img_L = self.uint2single(img_L)

            # HWC to CHW, numpy to tensor
            img_L = self.single2tensor3(img_L)
            img_L = img_L.unsqueeze(0)
            with torch.no_grad():
                output = self.crop_predict(img_L)

            restored = torch.clip(output, 0, 1)

            restored = restored.numpy()
            restored = restored.transpose(0, 2, 3, 1)
            restored = restored[0]
            restored = restored * 255
            restored = restored.astype(np.uint8)

            cv2.imwrite(restoration_save_path,
                        cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

    def get_images(self, images_path):
        if images_path is None:
            return []
        image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
        return image_files

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="picture_restoration")
    parser.add_argument("--input_path", type=str, default="./4kpictures/inputs", help="定义输入路径")
    parser.add_argument("--output_path", type=str, default="./4kpictures/outputs", help="定义输出路径")
    parser.add_argument("--weight_path", type=str, default="./models/NAFNet-REDS-width64.pdparams", help="定义权重所在路径")
    args = parser.parse_args()

    # 定义去模糊类
    croppredictor = NAFNetDeblurer(args.output_path, args.weight_path)
    # 执行预测
    croppredictor.run(images_path=args.input_path)
