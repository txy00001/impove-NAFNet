# 导入必要的库
import os
from basicsr.utils.create_lmdb import make_lmdb_from_imgs

def create_lmdb_for_reds():
    # 设置文件夹路径和LMDB路径
    folder_path = '/mnt/P40_NFS/20_Research/10_公共数据集/40_DeBlur/REDS/train/train_blur'
    lmdb_path = '/mnt/P40_NFS/20_Research/10_公共数据集/40_DeBlur/REDS/train/train_sharp.lmdb'

    # 调用make_lmdb_from_imgs函数创建LMDB数据库
    img_path_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    keys = [file.split('.')[0] for file in os.listdir(folder_path)]  # 假设文件名格式为"image_name.png"
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

if __name__ == '__main__':
    create_lmdb_for_reds()
