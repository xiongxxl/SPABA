import os
import pandas as pd
import matplotlib.pyplot as plt
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_dir_dir=os.path.dirname(parent_dir)

data_files_smiles='data/result/statistics_reactive/double_70/df_ratio_multi_0.90_0.99_v02_20241215.xlsx'
foldername=os.path.join(parent_dir_dir, data_files_smiles)
df_error=pd.read_excel(foldername)
df_zero=pd.DataFrame()
df_one=pd.DataFrame()
df_two=pd.DataFrame()
df_three=pd.DataFrame()
df_four=pd.DataFrame()
df_five=pd.DataFrame()
df_zero['ratio']=df_error['ratio']
df_zero['0_sum']=df_error['0_number']
df_one['ratio'] = df_error['ratio']
df_one['1_sum'] = df_error['0_number']+df_error['1_number']
df_two['ratio'] = df_error['ratio']
df_two['2_sum'] = df_error['0_number']+df_error['1_number']+df_error['2_number']
df_three['ratio'] = df_error['ratio']
df_three['3_sum'] = df_error['0_number']+df_error['1_number']+df_error['2_number']+df_error['3_number']
df_four['ratio']  = df_error['ratio']
df_four['4_sum']  = df_error['0_number']+df_error['1_number']+df_error['2_number']+df_error['3_number']+df_error['4_number']
df_five['ratio']  = df_error['ratio']
df_five['5_sum']  = df_error['0_number']+df_error['1_number']+df_error['2_number']+df_error['3_number']+df_error['4_number']+df_error['5_number']
##error 0

x=df_zero['ratio']
y=df_zero['0_sum']
plt.plot(df_zero['ratio'], df_zero['0_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-0 attoms error", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-0 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-0 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()  # 显示图形

##error 1
x=df_one['ratio']
y=df_one['1_sum']
plt.plot(df_one['ratio'], df_one['1_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-1 attoms error", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-1 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-1 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()  # 显示图形

##error 2
x=df_two['ratio']
y=df_two['2_sum']
plt.plot(df_two['ratio'], df_two['2_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-2 attoms error", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-2 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-2 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()  # 显示图形

##error 3
import matplotlib.pyplot as plt
x=df_three['ratio']
y=df_three['3_sum']

plt.plot(df_three['ratio'], df_three['3_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-3 attoms error", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-3 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-3 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()  # 显示图形

##error 4
import matplotlib.pyplot as plt
x=df_four['ratio']
y=df_four['4_sum']

plt.plot(df_four['ratio'], df_four['4_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-4 attoms erro", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-4 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-4 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()


##error 5
import matplotlib.pyplot as plt
x=df_five['ratio']
y=df_five['5_sum']

plt.plot(df_five['ratio'], df_five['5_sum'], marker='o', linestyle='-', color='b')  # 画出x和y的关系，点的样式为'o'
plt.title("0-5 attoms erro", fontsize=16, fontweight='bold')
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
plt.ylabel("0-5 atoms error", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'0-5 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()

##fix singe value not sum

import matplotlib.pyplot as plt
import pandas as pd
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['0_number']
data=df_error
x_max = df_error.loc[df_error['0_number'].idxmax(), 'ratio']
y_max = df_error['0_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['0_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 0 atom error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("0_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 0 atom error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()


##1 error
import matplotlib.pyplot as plt
import pandas as pd
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['1_number']
data=df_error
x_max = df_error.loc[df_error['1_number'].idxmax(), 'ratio']
y_max = df_error['1_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['1_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 1 atom error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("1_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 1 atom error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()


##2
import matplotlib.pyplot as plt
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['2_number']
data=df_error
x_max = df_error.loc[df_error['2_number'].idxmax(), 'ratio']
y_max = df_error['2_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['2_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 2 atom error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("2_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 2 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()


##3
import matplotlib.pyplot as plt
import pandas as pd
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['3_number']
data=df_error
x_max = df_error.loc[df_error['0_number'].idxmax(), 'ratio']
y_max = df_error['3_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['3_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 3 atom error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("3_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 3 atoms error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()


##4
import matplotlib.pyplot as plt
import pandas as pd
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['4_number']
data=df_error
x_max = df_error.loc[df_error['4_number'].idxmax(), 'ratio']
y_max = df_error['4_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['4_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 4 atoms error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("4_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 4 atom error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()

##5
import matplotlib.pyplot as plt
import pandas as pd
# 创建包含两列数据的DataFrame
# data = pd.DataFrame({
#     'x': [1, 3],   # x坐标
#     'y': [4, 6]    # y坐标
# })
# 找到最大y值对应的x和y
x=df_error['ratio']
y=df_error['5_number']
data=df_error
x_max = df_error.loc[df_error['5_number'].idxmax(), 'ratio']
y_max = df_error['5_number'].max()
y_max=round(y_max,4)
# 绘制散点图（不连线）
plt.figure(figsize=(10, 8))
plt.scatter(df_error['ratio'], df_error['5_number'], color='blue', s=100, zorder=5)  # 's=100' 设置点的大小，'zorder' 确保点位于前面
# 设置标题并调整字体大小和加粗
plt.title("The ratio of 5 atoms error", fontsize=16, fontweight='bold')
# 设置横轴标签并调整字体大小和加粗
plt.xlabel("Ratio ", fontsize=14, fontweight='bold')
# 设置纵轴标签并调整字体大小和加粗
plt.ylabel("5_number", fontsize=14, fontweight='bold')
# 加粗坐标轴数值
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# 在最大值点显示坐标
plt.text(x_max, y_max, f'({x_max},{y_max})', fontsize=12, ha='left',va='top', color='black',fontweight='bold')

data_files_smiles='data/result/statistics_reactive/double_70/ratio_img'
ratio_img=os.path.join(parent_dir_dir, data_files_smiles)
file_name=f'The ratio of 5 atom error.jpg'
ratio_img_detail=os.path.join(ratio_img,file_name)
plt.savefig(ratio_img_detail)
plt.show()
