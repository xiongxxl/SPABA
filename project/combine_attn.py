import numpy as np
import os
def add_npy_files(folder1, folder2, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取两个文件夹中的文件列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 找出两个文件夹中都存在的文件
    common_files = files1 & files2

    for file in common_files:
        if file.endswith('.npy'):
            # 构建完整文件路径
            path1 = os.path.join(folder1, file)
            path2 = os.path.join(folder2, file)

            # 加载npy文件
            arr1 = np.load(path1)
            arr2 = np.load(path2)

            # 检查数组形状是否相同
            if arr1.shape != arr2.shape:
                print(f"警告: {file} 的形状不匹配 ({arr1.shape} vs {arr2.shape})，跳过此文件")
                continue

            # 相加数组
            # result = np.maximum(arr1,arr2)
            # result=np.vstack((arr1,arr2))  #纵向拼接
            result=np.hstack((arr1,arr2))  #横向拼接

            # 保存结果
            output_path = os.path.join(output_folder, file)
            np.save(output_path, result)
            print(f"已处理: {file}")

    print("处理完成!")

head='4_5'
attn='del'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sample_file_name_4_5=f'data/middle_attention/uspto_sample/{head}/npy/deep_attn_{attn}_{head}'
sample_file_4_5=os.path.join(parent_dir,sample_file_name_4_5)


head='7_7'
attn='del'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sample_file_name_7_7=f'data/middle_attention/uspto_sample/{head}/npy/deep_attn_{attn}_{head}'
sample_file_7_7=os.path.join(parent_dir,sample_file_name_7_7)


head='combine'
attn='del'
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sample_file_name_combine=f'data/middle_attention/uspto_sample/{head}/deep_attn_{attn}_{head}'
sample_file_combine=os.path.join(parent_dir,sample_file_name_combine)



# 使用示例
folder1 =sample_file_4_5  # 替换为第一个文件夹路径
folder2 = sample_file_7_7  # 替换为第二个文件夹路径
output_folder = sample_file_combine  # 替换为输出文件夹路径

add_npy_files(folder1, folder2, output_folder)