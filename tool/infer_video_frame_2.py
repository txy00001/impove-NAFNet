import torch
import numpy as np
import cv2
import os
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    model.test()
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    return sr_img

def process_video_frames(input_path, output_folder, model_path, frame_indices):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    opt = parse(model_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    success, frame = cap.read()
    frame_count = 0
    while success:
        # 按指定帧序列处理视频
        if frame_count in frame_indices:
            img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = img2tensor(img_input)
            output_frame = single_image_inference(model, inp)
            output_frame = (output_frame * 255).astype(np.uint8)  # 将输出转换回uint8格式
            output_path = os.path.join(output_folder, f"frame_{frame_count}.png")
            imwrite(output_frame, output_path)

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# 示例：仅处理第1、10、20帧
input_video_path = '/home/txy/code/NAFNet/demo/frame/1229_havefault.mp4'
output_folder = '/home/txy/code/NAFNet/demo_out_frames'
model_path = 'options/test/REDS/NAFNet-width64.yml'
frame_indices = [0, 9, 19]  # 假设我们想处理的是第1、10、20帧（索引从0开始）

process_video_frames(input_video_path, output_folder, model_path, frame_indices)