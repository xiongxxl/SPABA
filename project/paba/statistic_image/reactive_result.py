import matplotlib.pyplot as plt
import numpy as np
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
filename_predict=os.path.join(parent_dir,'data/result/statistics_supervision/uspto_1244/find_alpha')

## 4_5和7_7 result
categories = ['0', '2', '4']
values1 = [12, 132, 258]
values2 = [17, 147, 309]
# values3 = [0, 0, 1]
# 计算每组的总和
# total1 = sum(values1)
# total2 = sum(values2)
# total3 = sum(values3)
total1 = 1089
total2 = 1089
# total3 = 5

# 计算百分比
percentages1 = [v / total1 * 100 for v in values1]
percentages2 = [v / total2 * 100 for v in values2]
# percentages3 = [v / total3 * 100 for v in values3]

# 绘制柱状图
width = 0.2  # 每组柱状图的宽度
x = np.arange(len(categories))  # X轴的位置

fig, ax = plt.subplots()

ax.bar(x - width, percentages1, width, label='Single molecule',color='#C2B2D6')
ax.bar(x, percentages2, width, label='Double molecules',color='#FDC897')
# ax.bar(x + width, percentages3, width, label='Three molecules',color='#9DD79D')

# 添加标签
ax.set_xticks(x)
ax.set_xticklabels(categories)

# 添加百分比标签
for i in range(len(categories)):
    ax.text(x[i] - width, percentages1[i] + 0.35, f'{percentages1[i]:.2f}%', ha='center', va='bottom')
    ax.text(x[i], percentages2[i] + 0.35, f'{percentages2[i]:.2f}%', ha='center', va='bottom')
    # ax.text(x[i] + width, percentages3[i] + 0.35, f'{percentages3[i]:.2f}%', ha='center', va='bottom')


ax.legend(loc='upper left')
plt.xlabel('Atoms_error',fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)',fontsize=14, fontweight='bold')
plt.title(' Different numbers of reactants(4_5) ',fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
jpg_path=f'different_reactants_4_5.jpeg'
different_reactants_path=os.path.join(filename_predict,jpg_path)
plt.savefig(different_reactants_path)
plt.show()



#
# ### result of  different reactive molecules(4_5 head) ###
# categories = ['0', '2', '4']
# values1 = [1, 4, 5]
# values2 = [7, 28, 39]
# values3 = [0, 0, 1]
# # 计算每组的总和
# # total1 = sum(values1)
# # total2 = sum(values2)
# # total3 = sum(values3)
# total1 = 8
# total2 = 100
# total3 = 5
#
# # 计算百分比
# percentages1 = [v / total1 * 100 for v in values1]
# percentages2 = [v / total2 * 100 for v in values2]
# percentages3 = [v / total3 * 100 for v in values3]
#
# # 绘制柱状图
# width = 0.2  # 每组柱状图的宽度
# x = np.arange(len(categories))  # X轴的位置
#
# fig, ax = plt.subplots()
#
# ax.bar(x - width, percentages1, width, label='Single molecule',color='#C2B2D6')
# ax.bar(x, percentages2, width, label='Double molecules',color='#FDC897')
# ax.bar(x + width, percentages3, width, label='Three molecules',color='#9DD79D')
#
# # 添加标签
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
#
# # 添加百分比标签
# for i in range(len(categories)):
#     ax.text(x[i] - width, percentages1[i] + 0.35, f'{percentages1[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i], percentages2[i] + 0.35, f'{percentages2[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i] + width, percentages3[i] + 0.35, f'{percentages3[i]:.2f}%', ha='center', va='bottom')
#
#
# ax.legend(loc='upper left')
# plt.xlabel('Atoms_error',fontsize=14, fontweight='bold')
# plt.ylabel('Percentage (%)',fontsize=14, fontweight='bold')
# plt.title(' Different numbers of reactants(4_5) ',fontsize=16, fontweight='bold')
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# jpg_path=f'different_reactants_4_5.jpeg'
# different_reactants_path=os.path.join(filename_predict,jpg_path)
# plt.savefig(different_reactants_path)
# plt.show()
#
#
#
#
# ### result of  different reactive molecules(7_7 head) ###
# categories = ['0', '2', '4']
# values1 = [1, 2, 4]
# values2 = [0, 15, 32]
# values3 = [0, 0, 0]
# # 计算每组的总和
# # total1 = sum(values1)
# # total2 = sum(values2)
# # total3 = sum(values3)
# total1 = 8
# total2 = 100
# total3 = 5
#
# # 计算百分比
# percentages1 = [v / total1 * 100 for v in values1]
# percentages2 = [v / total2 * 100 for v in values2]
# percentages3 = [v / total3 * 100 for v in values3]
#
# # 绘制柱状图
# width = 0.2  # 每组柱状图的宽度
# x = np.arange(len(categories))  # X轴的位置
#
# fig, ax = plt.subplots()
#
# ax.bar(x - width, percentages1, width, label='Single molecule',color='#C2B2D6')
# ax.bar(x, percentages2, width, label='Double molecules',color='#FDC897')
# ax.bar(x + width, percentages3, width, label='Three molecules',color='#9DD79D')
#
# # 添加标签
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
#
# # 添加百分比标签
# for i in range(len(categories)):
#     ax.text(x[i] - width, percentages1[i] + 0.35, f'{percentages1[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i], percentages2[i] + 0.35, f'{percentages2[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i] + width, percentages3[i] + 0.35, f'{percentages3[i]:.2f}%', ha='center', va='bottom')
#
# ax.legend(loc='upper left')
# plt.xlabel('Atoms_error',fontsize=14, fontweight='bold')
# plt.ylabel('Percentage (%)',fontsize=14, fontweight='bold')
# plt.title(' Different numbers of reactants(7_7) ',fontsize=16, fontweight='bold')
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# jpg_path=f'different_reactants_7_7.jpeg'
# different_reactants_path=os.path.join(filename_predict,jpg_path)
# plt.savefig(different_reactants_path)
# plt.show()
#
#
#
#
# ### result of  three methods ###
# categories = ['0', '2', '4']
# # function =[3,20,27]
# # combine  =[4,30,48]
# # predict  =[4,26,40]
# values1 = [3,20,27]
# values2 = [7,37,54]
# values3 = [7,31,45]
# # 计算每组的总和
# # total1 = sum(values1)
# # total2 = sum(values2)
# # total3 = sum(values3)
# total1 = 100
# total2 = 100
# total3 = 100
#
# # 计算百分比
# percentages1 = [v / total1 * 100 for v in values1]
# percentages2 = [v / total2 * 100 for v in values2]
# percentages3 = [v / total3 * 100 for v in values3]
#
# # 绘制柱状图
# width = 0.2  # 每组柱状图的宽度
# x = np.arange(len(categories))  # X轴的位置
#
# fig, ax = plt.subplots()
#
# ax.bar(x - width, percentages1, width, label='Function',color='#C2B2D6')
# ax.bar(x, percentages2, width, label='Combine',color='#FDC897')
# ax.bar(x + width, percentages3, width, label='Predict',color='#9DD79D')
#
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
#
# # 添加百分比标签
# for i in range(len(categories)):
#     ax.text(x[i] - width, percentages1[i] +0.35, f'{percentages1[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i], percentages2[i] + 0.35, f'{percentages2[i]:.2f}%', ha='center', va='bottom')
#     ax.text(x[i] + width, percentages3[i] + 0.35, f'{percentages3[i]:.2f}%', ha='center', va='bottom')
#
# # 显示图例
# ax.legend(loc='upper left')
#
# # 加粗坐标轴数字
# # ax.tick_params(axis='both', which='major', labelsize=14, width=4)  # labelsize 调整数字大小，width 调整线条宽度
# # ax.tick_params(axis='both', which='minor', labelsize=10, width=2)  # minor ticks 如果有的话
# plt.xlabel('Atoms_error',fontsize=14, fontweight='bold')
# plt.ylabel('Percentage (%)',fontsize=14, fontweight='bold')
# plt.title('Result of three methods',fontsize=16, fontweight='bold')
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
#
# jpg_method_path=f'results of three methods.jpeg'
# three_method_path=os.path.join(filename_predict,jpg_method_path)
# plt.savefig(three_method_path)
# plt.show()





# ## results of three method in 100 sample###
# atoms_error = ['0', '2', '4']
# function =[3,20,27]
# combine  =[4,30,48]
# predict  =[4,26,40]
#
# bar_width = 0.2
# index = np.arange(len(atoms_error))
#
# plt.bar(index - bar_width, function, bar_width, label='function', color='#9DD79D')
# plt.bar(index, combine, bar_width, label='combine', color='#FDC897')
# plt.bar(index + bar_width, predict, bar_width, label='predict', color='#C2B2D6')
#
# plt.xlabel('Atoms_error',fontsize=14, fontweight='bold')
# plt.ylabel('Number of sample',fontsize=14, fontweight='bold')
# plt.title('Result of three method',fontsize=16, fontweight='bold')
# plt.xticks(index, atoms_error)
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# plt.legend()
# plt.tight_layout()
# plt.show()




