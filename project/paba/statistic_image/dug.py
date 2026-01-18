#

# from collections import Counter
#
# # 数据
# data = ['4_5', '4_5', '4_6', '7_0', '6_7', '2_1', '5_0', '4_5', '7_6', '4_5', '7_7', '4_5', '7_6', '3_3', '7_7', '4_5', '4_6', '5_7', '7_7', '4_5', '7_7', '1_0', '4_5', '3_4', '3_6', '4_5', '4_1', '3_1', '2_1', '4_5', '4_1', '7_7', '7_4', '3_4', '1_3', '4_1', '7_7', '2_0', '6_2', '4_5', '6_5', '5_1', '3_4', '3_7', '1_3', '7_5', '7_6', '4_5', '7_7', '3_4', '7_7', '3_6', '4_5', '2_0', '3_2', '2_1', '7_6', '3_3', '2_1', '5_6', '7_0', '6_7', '4_6', '1_0', '3_6', '2_5', '2_6', '2_0', '4_5', '4_5', '7_6', '4_5', '7_7', '5_2', '6_1', '4_5', '5_5', '4_1', '7_7', '5_7', '3_3', '4_2', '5_6', '7_7', '7_7', '7_1', '4_5', '3_7', '6_7', '7_7', '2_1', '7_7', '4_5', '7_7', '4_5', '7_7', '4_5', '3_4', '7_6', '7_5', '7_7', '3_3', '4_5', '4_5', '7_6', '7_5', '3_4', '4_5', '6_3', '4_4', '7_6', '4_5', '4_6', '1_0', '2_4', '4_5', '7_7', '4_5', '7_7', '1_0', '7_7', '7_7', '4_6', '5_4', '7_7', '1_4', '4_5', '7_1', '2_0', '7_6', '7_5', '3_4', '7_6', '2_0', '2_5', '1_4', '1_0', '7_7', '4_5', '2_0', '4_1', '1_0', '4_5', '7_7', '1_1', '6_2', '6_1', '7_7', '4_5', '4_5', '7_7', '2_4', '4_5', '4_6', '3_3', '5_2', '3_6', '7_7', '3_7', '7_6', '4_5', '3_4', '7_7', '4_5', '7_6', '7_1', '7_5', '6_1', '7_6', '3_7', '4_5', '5_5', '7_7', '1_0', '2_1', '7_7', '6_5', '3_6', '5_5', '6_2', '4_5', '7_7', '2_5', '4_5', '7_7']
#
# # 统计每个样本出现的次数
# counter = Counter(data)
#
# # 按出现次数从高到低排序
# sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
#
# # 输出结果
# for item, count in sorted_counts:
#     print(f"{item}: {count}次")


import matplotlib.pyplot as plt
# 数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]
# 创建柱状图
plt.bar(categories, values, color='#b4e0f6')
# 添加标题和标签
plt.title('柱状图示例')
plt.xlabel('类别')
plt.ylabel('值')
# 显示图表
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 示例数据（三组数组）
# categories = ['A', 'B', 'C']
# values1 = [1, 4, 5]
# values2 = [30, 30, 25]
# values3 = [0, 0, 1]
#
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
# ax.bar(x - width, percentages1, width, label='Group 1')
# ax.bar(x, percentages2, width, label='Group 2')
# ax.bar(x + width, percentages3, width, label='Group 3')
#
# # 添加标签
# ax.set_ylabel('Percentage (%)')
# ax.set_title('Percentage Bar Chart for Three Groups')
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
#
# # 添加百分比标签
# for i in range(len(categories)):
#     ax.text(x[i] - width, percentages1[i] + 1, f'{percentages1[i]:.1f}%', ha='center', va='bottom')
#     ax.text(x[i], percentages2[i] + 1, f'{percentages2[i]:.1f}%', ha='center', va='bottom')
#     ax.text(x[i] + width, percentages3[i] + 1, f'{percentages3[i]:.1f}%', ha='center', va='bottom')
#
# # 显示图例
# ax.legend()
# plt.show()




# import matplotlib.pyplot as plt
# # 示例数据
# categories = ['A', 'B', 'C', 'D']
# values = [25, 35, 20, 20]
# # 计算百分比
# total = sum(values)
# percentages = [v / total * 100 for v in values]
# # 绘制柱状图
# plt.bar(categories, percentages)
# # 添加标签
# plt.ylabel('Percentage (%)')
# plt.title('Percentage Bar Chart')
# # 在柱子上方显示百分比
# for i, v in enumerate(percentages):
#     plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
# plt.show()




# import matplotlib.pyplot as plt
# # 假设x和y是你的数据点
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]
# # 绘制数据点
# plt.plot(x, y, 'bo')
# # 为每个数据点添加标注，并通过xytext设置偏移量
# for i in range(len(x)):
#     plt.annotate(f'({x[i]},{y[i]})', (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
# plt.show()

# import matplotlib.pyplot as plt
# # 数据
# ratio = [0.9, 0.901, 0.902, 0.903, 0.904, 0.905, 0.906, 0.907, 0.908, 0.909,
#          0.91, 0.911, 0.912, 0.913, 0.914, 0.915, 0.916, 0.917, 0.918, 0.919]
# number = [0, 19, 18, 17, 17, 17, 17, 17, 17, 17,
#           17, 17, 17, 17, 17, 17, 16, 16, 16, 9]
# heads = [0.5] * len(ratio)
# # 创建绘图
# plt.figure(figsize=(10, 6))
# plt.scatter(ratio, number, label='Data Points', c='blue', alpha=0.7)
# # 给每个点添加注释 (head值)
# for i, head in enumerate(heads):
#     plt.text(ratio[i], number[i], f"{head}", fontsize=9, ha='right')
# # 添加标签和标题
# plt.xlabel('Ratio')
# plt.ylabel('Number')
# plt.title('Scatter Plot of Ratio vs Number with Head Annotations')
# plt.grid(alpha=0.3)
# # 显示绘图
# plt.show()