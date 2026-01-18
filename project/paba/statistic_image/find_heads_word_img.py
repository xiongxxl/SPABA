import math
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from wordcloud import WordCloud
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_reactive/double_para/find_heads')

# 数据

# df_single_rwo=pd.read_excel(os.path.join(input_files_smiles,'df_single_row_find_head_0.99.xlsx'))
#
# head_combined_re=df_single_rwo.loc[:,'head_combined_re']

# head_combined_re = ['4_5', '4_5', '4_6', '7_0', '6_7', '2_1', '5_0', '4_5', '7_6', '4_5', '7_7', '4_5', '7_6', '3_3', '7_7', '4_5', '4_6', '5_7', '7_7', '4_5', '7_7', '1_0', '4_5', '3_4', '3_6', '4_5', '4_1', '3_1', '2_1', '4_5', '4_1', '7_7', '7_4', '3_4', '1_3', '4_1', '7_7', '2_0', '6_2', '4_5', '6_5', '5_1', '3_4', '3_7', '1_3', '7_5', '7_6', '4_5', '7_7', '3_4', '7_7', '3_6', '4_5', '2_0', '3_2', '2_1', '7_6', '3_3', '2_1', '5_6', '7_0', '6_7', '4_6', '1_0', '3_6', '2_5', '2_6', '2_0', '4_5', '4_5', '7_6', '4_5', '7_7', '5_2', '6_1', '4_5', '5_5', '4_1', '7_7', '5_7', '3_3', '4_2', '5_6', '7_7', '7_7', '7_1', '4_5', '3_7', '6_7', '7_7', '2_1', '7_7', '4_5', '7_7', '4_5', '7_7', '4_5', '3_4', '7_6', '7_5', '7_7', '3_3', '4_5', '4_5', '7_6', '7_5', '3_4', '4_5', '6_3', '4_4', '7_6', '4_5', '4_6', '1_0', '2_4', '4_5', '7_7', '4_5', '7_7', '1_0', '7_7', '7_7', '4_6', '5_4', '7_7', '1_4', '4_5', '7_1', '2_0', '7_6', '7_5', '3_4', '7_6', '2_0', '2_5', '1_4', '1_0', '7_7', '4_5', '2_0', '4_1', '1_0', '4_5', '7_7', '1_1', '6_2', '6_1', '7_7', '4_5', '4_5', '7_7', '2_4', '4_5', '4_6', '3_3', '5_2', '3_6', '7_7', '3_7', '7_6', '4_5', '3_4', '7_7', '4_5', '7_6', '7_1', '7_5', '6_1', '7_6', '3_7', '4_5', '5_5', '7_7', '1_0', '2_1', '7_7', '6_5', '3_6', '5_5', '6_2', '4_5', '7_7', '2_5', '4_5', '7_7']
# counter = Counter(head_combined_re)
# sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=False)
#
# # 输出结果
# for item, count in sorted_counts:
#     print(f"{item}: {count}")


# histogram
# ## 1
# data={
# '5_0': 1,
# '3_1': 1,
# '7_4': 1,
# '5_1': 1,
# '3_2': 1,
# '2_6': 1,
# '4_2': 1,
# '6_3': 1,
# '4_4': 1,
# '5_4': 1,
# '1_1': 1,
# '7_0': 2,
# '5_7': 2,
# '1_3': 2,
# '6_5': 2,
# '5_6': 2,
# '5_2': 2,
# '2_4': 2,
# '1_4': 2,
# '6_7': 3,
# '6_2': 3,
# '2_5': 3,
# '6_1': 3,
# '5_5': 3,
# '7_1': 3,
# '3_7': 4,
# '3_3': 5,
# '3_6': 5,
# '4_1': 5,
# '7_5': 5,
# '4_6': 6,
# '2_1': 6,
# '2_0': 6,
# '1_0': 7,
# '3_4': 8,
# '7_6': 13,
# '7_7': 32,
# '4_5': 38,
#         }
#
# data = {key: value for key, value in data.items()}
# group_data = list(data.values())
# group_names = list(data.keys())
# group_mean = np.mean(group_data)
# fig, ax = plt.subplots(figsize=(8,8))
# ax.barh(group_names,group_data,color='#b4e0f6')
# ax.set_xlabel('Number', fontsize=16, fontweight='bold')
# ax.set_ylabel('head_axis', fontsize=16, fontweight='bold')
# ax.set_title('Number of heads of containing reactive site', fontsize=16, fontweight='bold')
# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.set(xlim=[0, 40])
# ax.tick_params(axis='both', which='major', labelsize=12)  # 设置刻度标签字体大小
# for tick in ax.get_xticklabels() + ax.get_yticklabels():
#     tick.set_fontweight('bold')  # 设置数字加粗
# plt.ylim(-1, len(group_names) )  # 调整 y 轴范围
# plt.margins(y=0)  # 移除多余边距
# plt.tight_layout()
# fragment_img=f'The top abundant 1-atom fragments'  #save functional group path
# fold_histogram=os.path.join(input_files_smiles,'histogram')
# statistics_fragment_path= os.path.join(fold_histogram, fragment_img)
# plt.savefig(statistics_fragment_path)
# plt.show()
# plt.close()
#
# # word img

##2
input_files_smiles = os.path.join(parent_dir, 'data/result/statistics_reactive/double_para/find_heads')

df_single_rwo=pd.read_excel(os.path.join(input_files_smiles,'df_all_head_frequent_0.99.xlsx'))
word_freq = dict(zip(df_single_rwo['head'], df_single_rwo['frequent']))
wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Head_axis of containing reactive site",fontsize=16, fontweight='bold')
plt.axis('off')

file_word=f'word_img'
word_img_folder = os.path.join(input_files_smiles,file_word)
img_name = f'word_image of 1 atom.jpeg'
img_name_detail=os.path.join(word_img_folder,img_name)
plt.savefig(img_name_detail, bbox_inches='tight')
plt.show()

##3
# histogram
## 1

import matplotlib.pyplot as plt
# 数据
categories = ['0', '1', '2', '3', '4','5']
values = [16, 27, 32, 26, 24,15]
# 创建柱状图
plt.bar(categories, values, color='#b4e0f6')
# 添加标题和标签

plt.title("Number of molecules of containing reactive site ", fontsize=12, fontweight='bold')
plt.xlabel("Atoms_error", fontsize=14, fontweight='bold')
plt.ylabel("Number", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

file_name=f'statistic_result.jpeg'
ratio_img_detail=os.path.join(input_files_smiles,file_name)
plt.savefig(ratio_img_detail)
plt.show()




