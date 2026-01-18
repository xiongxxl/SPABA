
import os
import pandas as pd
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_dir_dir=os.path.dirname(parent_dir)

####  pie chart ###
# plt.rcParams['figure.figsize'] = (6,6)
# labels = [(4,4),(5,4),(4,5)]
# x = [54,33,13]
# plt.title("Atention location ratio",fontsize=16, fontweight='bold')
# plt.pie(x,labels=labels,autopct='%.2f%%',textprops={'fontsize': 12})
# plt.tight_layout()
# plt.savefig('Atention location ratio.png')
# plt.show()

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (6,6)
# labels = [0,1,2,'NA']
# x = [8,25,26,41]
# plt.title("Atoms error ratio",fontsize=16, fontweight='bold')
# plt.pie(x,labels=labels,autopct='%.2f%%')
# plt.tight_layout()
# plt.savefig('Atoms error ratio.png')
# plt.show()

### bar chart  ###

#percentage
# import matplotlib.pyplot as plt
# import numpy as np
# labels = ['0', '2', '4']
# percentages = [3,21,33]
# # # 不同颜色
#  #colors = ['blue', 'orange', 'brown', 'purple','green']
# colors='#b4e0f6'
#  # 绘制柱状图
# plt.bar(labels, percentages, color=colors)
#  # 添加标题和标签
# plt.title('4_5 head in double molecules',fontsize=16, fontweight='bold')
# plt.xlabel('beta',fontsize=16, fontweight='bold')
# plt.ylabel('percentage',fontsize=16, fontweight='bold')
# #
# # # 添加百分比文本
# for i, percentage in enumerate(percentages):
#     plt.text(i, percentage + 1, f'{percentage:.1f}%', ha='center')
# plt.ylim(0, 110)  # 设置 y 轴范围
# plt.savefig('4_5_double.png')
# plt.show()

#
# import matplotlib.pyplot as plt
# # 数据
# categories = ['0', '2', '4']
# values = [1,4,5]
# # 计算总和
# total = 8
# # 计算百分比
# percentages = [value / total * 100 for value in values]
# # 创建柱状图
# fig, ax = plt.subplots()
# colors='#b4e0f6'
# bars = ax.bar(categories, values,color=colors)
# # 为每个柱子添加百分比标签
# for bar, percentage in zip(bars, percentages):
#     height = bar.get_height()  # 获取柱子的高度
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,  # x位置（柱子中间）
#         height ,  # y位置，略高于柱子的顶部
#         f'{percentage:.1f}%',  # 显示的文本
#         ha='center',  # 水平对齐方式
#         va='bottom',  # 垂直对齐方式
#         fontsize=10  # 字体大小
#     )
# # 添加标题和标签
# plt.title(' 4_5 head in single molecules results',fontsize=16, fontweight='bold')
# plt.xlabel('beta',fontsize=16, fontweight='bold')
# plt.ylabel('percentage',fontsize=16, fontweight='bold')
# plt.savefig('4_5_single molecules.png')
# plt.show()


