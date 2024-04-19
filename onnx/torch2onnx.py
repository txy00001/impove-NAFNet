
import os
import torch
import onnxruntime
from onnxruntime.datasets import get_example
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img, imwrite
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel



def model_to_device(net):
    """Model to device. It also warps models with DistributedDataParallel
    or DataParallel.

    Args:
        net (nn.Module)
    """
    # opt = parse_options(is_train=False)
    num_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if num_gpu != 0 else 'cpu')
    
    net = net.to(device)
    return net


def print_different_keys_loading(crt_net, load_net, strict=True):
    """Print keys with differnet name or different size when loading models.

    1. Print keys with differnet names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    if isinstance(crt_net, (DataParallel, DistributedDataParallel)):
        crt_net = crt_net.module
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())
    
    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                load_net[k + '.ignore'] = load_net.pop(k)


def main():
    
    width = 64
    img_channel = 3
    enc_blk_nums = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blk_nums = [1, 1, 1, 1]
    net_g = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
    # net_g = model_to_device(net_g)
    net_g = net_g.to("cuda")
    
    # load模型训练好的权重
    load_path = r"E:\work\NAFNet-main\experiments\pretrained_models\NAFNET_SAISI-width64-95000.pth"
    param_key = 'params'
    load_net = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    print(' load net keys', load_net.keys)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    print_different_keys_loading(net_g, load_net, strict=True)
    net_g.load_state_dict(load_net, strict=True)
    
    # dummy_input可以随机设置一个tensor,用于测试
    # dummy_input = torch.randn(1, 3, 280, 280, device="cuda")
    
    # 下面使用的原始图像经过变换变成dummy_input，上面随机生成的也可以
    # 用于测试和模型输入的图像，这里要注意的是图片的resize，后面转为onnx后模型就固定大小输入，不是动态的
    img_path = r"E:\work\NAFNet-main\datasets\caisi\test\Snipaste_2022-07-13_16-06-58.png"
    # 模型输出结果的图像路径
    output_path = r"E:\work\NAFNet-main\datasets\caisi\test\pp.png"
    file_client = FileClient('disk')
    img_bytes = file_client.get(img_path, None)
    img = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(dim=0)
    dummy_input = img.cuda()
    
    # 原始预测的结果可以和打包以后的onnx预测结果做对比
    pred = net_g(dummy_input)
    sr_img = tensor2img(pred)
    imwrite(sr_img, output_path)
    
    
    # 转为onnx及模型保存路径
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]
    onnx_path = "E:\\work\\NAFNet-main\\NAFNet.onnx"
    torch.onnx.export(net_g, dummy_input, onnx_path, verbose=True,
                      input_names=input_names, output_names=output_names, opset_version=11)
    
    
    # 使用onnx做验证及onnx结果保存路径
    output_onnx_path = r"E:\work\NAFNet-main\datasets\caisi\test\onnx_pp.png"
    onnx_model_path = "E:\\work\\NAFNet-main\\NAFNet.onnx"
    example_model = get_example(onnx_model_path)
    sess = onnxruntime.InferenceSession(example_model, providers=['CUDAExecutionProvider'])
    imgs = img.cpu().numpy()
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    onnx_out = sess.run(None, {input_names[0]: imgs})
    out_onnx_tensor = torch.from_numpy(onnx_out[0])
    sr_onnx_img = tensor2img(out_onnx_tensor)
    imwrite(sr_onnx_img, output_onnx_path)
    
    
if __name__ == '__main__':
    main()




