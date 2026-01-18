import os
from PIL import Image

def syn_1_jpg(image_folder,output_image):
    #input 64 img
    #output synthesis 1 imge
    #image_folder = './result/attention/first/img/CC(C)(C)c1cc(O)c(O)c(C(C)(C)C)c1'
    #output_image = 'final_image.jpg'

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort()  # 确保顺序一致

    # 假设所有图片的尺寸相同
    image_width, image_height = Image.open(image_files[0]).size
    grid_size = 8  # 8x8 网格

    # 创建空白的画布
    final_image = Image.new('RGB', (grid_size * image_width, grid_size * image_height))

    # 将每张图片粘贴到画布上
    for index, image_file in enumerate(image_files):
        image = Image.open(image_file)
        x = (index % grid_size) * image_width
        y = (index // grid_size) * image_height
        final_image.paste(image, (x, y))

    # 保存最终的拼接图片
    final_image.save(output_image)
    #print(f'Final image saved as {output_image}')
    return final_image

def syn_1_png(image_folder,output_image):
    #input 64 img
    #output synthesis 1 imge
    #image_folder = './result/attention/first/img/CC(C)(C)c1cc(O)c(O)c(C(C)(C)C)c1'
    #output_image = 'final_image.jpg'

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort()  # 确保顺序一致

    # 假设所有图片的尺寸相同
    image_width, image_height = Image.open(image_files[0]).size
    grid_size = 8  # 8x8 网格

    # 创建空白的画布
    final_image = Image.new('RGB', (grid_size * image_width, grid_size * image_height))

    # 将每张图片粘贴到画布上
    for index, image_file in enumerate(image_files):
        image = Image.open(image_file)
        x = (index % grid_size) * image_width
        y = (index // grid_size) * image_height
        final_image.paste(image, (x, y))

    # 保存最终的拼接图片
    final_image.save(output_image)
    #print(f'Final image saved as {output_image}')
    return final_image

def syn_1_jpeg(image_folder,output_image):
    #input 64 img
    #output synthesis 1 imge
    #image_folder = './result/attention/first/img/CC(C)(C)c1cc(O)c(O)c(C(C)(C)C)c1'
    #output_image = 'final_image.jpg'

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpeg')]
    image_files.sort()  # 确保顺序一致

    # 假设所有图片的尺寸相同
    image_width, image_height = Image.open(image_files[0]).size
    grid_size = 8  # 8x8 网格

    # 创建空白的画布
    final_image = Image.new('RGB', (grid_size * image_width, grid_size * image_height))

    # 将每张图片粘贴到画布上
    for index, image_file in enumerate(image_files):
        image = Image.open(image_file)
        x = (index % grid_size) * image_width
        y = (index // grid_size) * image_height
        final_image.paste(image, (x, y))

    # 保存最终的拼接图片
    final_image.save(output_image)
    #print(f'Final image saved as {output_image}')
    return final_image

if __name__ == "__main__":

    image_folder = './result/attention/first/img/CC(C)(C)c1cc(O)c(O)c(C(C)(C)C)c1'
    output_image = 'final_image.jpg'
