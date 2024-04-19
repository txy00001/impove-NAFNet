from PIL import Image
 
def resize_image(input_image_path, output_image_path, size=(416, 416)):
   """
   调整图片大小到指定尺寸并保存。
 
   参数:
   input_image_path: 输入图片的路径。
   output_image_path: 输出图片的保存路径。
   size: 调整后的图片尺寸 (width, height)。
   """
   # 打开图片
   image = Image.open(input_image_path)
   
   # 调整图片大小
   resized_image = image.resize(size, Image.BILINEAR)
   
   # 保存调整后的图片
   resized_image.save(output_image_path)
 
# 使用示例
input_image_path = 'pic/00000001.png'  # 输入图片路径
output_image_path = 'pic/00000001_new.png'  # 输出图片路径
resize_image(input_image_path, output_image_path)