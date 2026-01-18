# import os
# import matplotlib.pyplot as plt
import numpy as np
#
# import os
# # 指定要读取的文件夹路径
# folder_path = './result/8_npy'
# # 使用 os.walk() 递归遍历目录及其子目录
# items=os.listdir(folder_path)
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         print(os.path.join(root, file))
#         attn_twodim=np.load(os.path.join(root, file))
#         file_name,_=os.path.splitext(file)
#         plt.imshow(attn_twodim, cmap='Greys')
#         plt.xlabel(file_name)
#         plt.savefig(file_name)
#         plt.show()

i=1
while i<7:
    i=i+1
    print('i',i)




#p=np.load("attn_twodim_array.npy")
#A=p