# 定义元组列表

def divide_by_dot_criterion(dot_criterion,functional_arrays):
    # 初始化两个列表
    less_dot_criterion = []
    greater_equal_dot_criterion= []
    # 遍历每个元组
    for t in functional_arrays:
        # 对元组中的每个元素进行判断
        if all(x < dot_criterion for x in t):  # 所有元素小于8
            less_dot_criterion.append(t)
        else:  # 含有元素大于或等于8
            greater_equal_dot_criterion.append(t)
    return less_dot_criterion,greater_equal_dot_criterion

if __name__ == "__main__":
    dot_criterion=8
    functional_arrays=((9, 10), (4,), (8,), (0,), (3, 4), (7, 8))
    less_dot_criterion,greater_equal_dot_criterion=divide_by_dot_criterion(dot_criterion,functional_arrays)

    # 输出分类结果
    print("小于8的元组:", less_dot_criterion)
    print("大于等于8的元组:", greater_equal_dot_criterion)