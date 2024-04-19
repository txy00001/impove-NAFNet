import torch
import numpy as np
import cv2
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

def process_video(input_path, output_path, model_path, start_frame=None, end_frame=None):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    opt = parse(model_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame is not None and frame_count < start_frame:
            frame_count += 1
            continue

        if end_frame is not None and frame_count > end_frame:
            break

        img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将输入帧转换为 RGB
        inp = img2tensor(img_input)
        output_frame = single_image_inference(model, inp)
        out.write(output_frame)  # 不再进行颜色转换

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video_path = '/home/txy/code/NAFNet/video/demo.mp4'
output_video_path = '/home/txy/code/NAFNet/demo_output/out_4.mp4'
model_path = 'options/test/REDS/NAFNet-width64.yml'

start_frame = 10
end_frame = 30

process_video(input_video_path, output_video_path, model_path, start_frame, end_frame)


