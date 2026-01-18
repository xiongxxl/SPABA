# import torch
# import torch.nn as nn
#
#
# network 2FCC
#
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         self.fc1 = nn.Linear(input_size * input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.view(x.size(0), -1)           # Flatten to [batch, 512*512]
#         x = self.relu(self.fc1(x))          # -> [batch, hidden_size]
#         x = self.dropout(x)
#         x = self.fc2(x)                     # -> [batch, output_size]
#         return x


#
# # ## 4FC
# #
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         self.fc1 = nn.Linear(input_size * input_size, 2048)
#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc3 = nn.Linear(1024, 768)
#         self.fc4 = nn.Linear(768, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.view(x.size(0), -1)       # Flatten to [batch, 262144]
#         x = self.relu(self.fc1(x))      # -> [batch, 2048]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc2(x))      # -> [batch, 1024]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc3(x))      # -> [batch, 768]
#         x = self.dropout(x)
#
#         x = self.fc4(x)                 # -> [batch, 512]
#         return x


# ### 6FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         self.fc1 = nn.Linear(input_size * input_size, 4096)
#         self.fc2 = nn.Linear(4096, 2048)
#         self.fc3 = nn.Linear(2048, 1024)
#         self.fc4 = nn.Linear(1024, 768)
#         self.fc5 = nn.Linear(768, 512)
#         self.fc6 = nn.Linear(512, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.view(x.size(0), -1)       # Flatten to [batch, 262144]
#         x = self.relu(self.fc1(x))      # -> [batch, 4096]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc2(x))      # -> [batch, 2048]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc3(x))      # -> [batch, 1024]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc4(x))      # -> [batch, 768]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc5(x))      # -> [batch, 512]
#         x = self.dropout(x)
#
#         x = self.fc6(x)                 # -> [batch, 512]
#         return x

# ##8FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         self.fc1 = nn.Linear(input_size * input_size, 8192)  # 扩展初始容量
#         self.fc2 = nn.Linear(8192, 4096)
#         self.fc3 = nn.Linear(4096, 2048)
#         self.fc4 = nn.Linear(2048, 1024)
#         self.fc5 = nn.Linear(1024, 768)
#         self.fc6 = nn.Linear(768, 512)
#         self.fc7 = nn.Linear(512, 512)
#         self.fc8 = nn.Linear(512, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.view(x.size(0), -1)          # Flatten: [batch, 262144]
#         x = self.relu(self.fc1(x))         # -> [batch, 8192]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc2(x))         # -> [batch, 4096]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc3(x))         # -> [batch, 2048]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc4(x))         # -> [batch, 1024]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc5(x))         # -> [batch, 768]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc6(x))         # -> [batch, 512]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc7(x))         # -> [batch, 512]
#         x = self.dropout(x)
#
#         x = self.fc8(x)                    # -> [batch, output_size]
#         return x


#
# # network 2CNN+2FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         self.fc1 = nn.Linear(hidden_size * (input_size // 4), hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 128]
#         x = self.relu(self.fc1(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.fc2(x)                   # -> [batch, output_size]
#
#         return x


# ##network 2CNN+3FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # 经过两次 MaxPool1d(2)，长度从 512 -> 256 -> 128
#         flattened_size = hidden_size * (input_size // 4)
#
#         # 3 Fully Connected layers
#         self.fc1 = nn.Linear(flattened_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 128]
#
#         x = self.relu(self.fc1(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.fc3(x)                   # -> [batch, output_size]
#
#         return x



# #####3NN+2FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         # 三个卷积层
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # 经过3次卷积 + 2次池化，序列长度从 512 -> 256 -> 128
#         flattened_size = hidden_size * (input_size // 4)
#
#         # 两个全连接层
#         self.fc1 = nn.Linear(flattened_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # -> [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.relu(self.conv3(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 128]
#
#         x = self.relu(self.fc1(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.fc2(x)                   # -> [batch, output_size]
#
#         return x



# ##network 2CNN+3FC
#
# import torch
# import torch.nn as nn
#
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # CNN 输出大小：hidden_size × (input_size // 4)
#         flattened_size = hidden_size * (input_size // 4)
#
#         # 3层全连接层
#         self.fc1 = nn.Linear(flattened_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # 输入形状: [batch, 512, 512]
#         x = x.permute(0, 2, 1)  # -> [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))  # -> [batch, hidden_size, 512]
#         x = self.pool(x)  # -> [batch, hidden_size, 256]
#
#         x = self.relu(self.conv2(x))  # -> [batch, hidden_size, 256]
#         x = self.pool(x)  # -> [batch, hidden_size, 128]
#
#         x = x.flatten(start_dim=1)  # -> [batch, hidden_size * 128]
#         x = self.relu(self.fc1(x))  # -> [batch, hidden_size]
#         x = self.dropout(x)
#
#         x = self.relu(self.fc2(x))  # -> [batch, hidden_size // 2]
#         x = self.dropout(x)
#
#         x = self.fc3(x)  # -> [batch, output_size]
#         return x





# # network 2CNN+2FC+64 inputszie
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=64, hidden_size=128, output_size=64):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         self.fc1 = nn.Linear(hidden_size * (input_size // 4), hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 128]
#         x = self.relu(self.fc1(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.fc2(x)                   # -> [batch, output_size]
#
#         return x

# # network 3CNN+2FC+64 inputszie
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=64, hidden_size=128, output_size=64):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # 原本是 input_size // 4，现在增加了一个pooling，所以是 // 8
#         self.fc1 = nn.Linear(hidden_size * (input_size // 8), hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # 输入 x: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, channels=512, seq_len=512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = self.relu(self.conv3(x))      # -> [batch, hidden, 128]
#         x = self.pool(x)                  # -> [batch, hidden, 64]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 64]
#         x = self.relu(self.fc1(x))        # -> [batch, hidden]
#         x = self.dropout(x)
#         x = self.fc2(x)                   # -> [batch, output_size]
#
#         return x





# ## network 2CNN+2FC+BN
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         # CNN 部分
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#
#         self.pool = nn.MaxPool1d(2)
#
#         # FC 部分
#         self.fc1 = nn.Linear(hidden_size * (input_size // 4), hidden_size)
#         self.bn_fc1 = nn.BatchNorm1d(hidden_size)
#
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         # 激活 & Dropout
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # 输入: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, channels=512, seq_len=512]
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.pool(x)                   # -> [batch, hidden, 256]
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.pool(x)                   # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)         # -> [batch, hidden * 128]
#
#         x = self.fc1(x)
#         x = self.bn_fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.fc2(x)                    # -> [batch, output_size=512]
#
#         return x

# #2CN+6FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # 经过两次MaxPool1d(2)，原始长度由512 -> 256 -> 128
#         flattened_size = hidden_size * (input_size // 4)
#
#         # 6 Fully Connected Layers
#         self.fc1 = nn.Linear(flattened_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, hidden_size)
#         self.fc4 = nn.Linear(hidden_size, hidden_size)
#         self.fc5 = nn.Linear(hidden_size, hidden_size)
#         self.fc6 = nn.Linear(hidden_size, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x shape: [batch, 512, 512]
#         x = x.permute(0, 2, 1)             # [batch, 512, 512] -> [batch, 512, 512]
#
#         x = self.relu(self.conv1(x))      # -> [batch, hidden, 512]
#         x = self.pool(x)                  # -> [batch, hidden, 256]
#         x = self.relu(self.conv2(x))      # -> [batch, hidden, 256]
#         x = self.pool(x)                  # -> [batch, hidden, 128]
#
#         x = x.flatten(start_dim=1)        # -> [batch, hidden * 128]
#
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc4(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc5(x))
#         x = self.dropout(x)
#         x = self.fc6(x)                   # -> [batch, output_size]
#
#         return x



# #network 5CNN+3FC
# import torch
# import torch.nn as nn
#
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         # 多层卷积
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
#
#         self.pool = nn.MaxPool1d(2)
#
#         # 计算经过 3 次池化后的序列长度: 512 → 256 → 128 → 64
#         self.flatten_size = hidden_size * (input_size // (2 ** 3))  # 512 // 8 = 64 → 1024 * 64
#
#         # 更深的全连接层
#         self.fc1 = nn.Linear(self.flatten_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x: [batch, 512, 512] → [batch, 512, 512]
#         x = x.permute(0, 2, 1)
#
#         x = self.relu(self.conv1(x))   # [B, 1024, 512]
#         x = self.pool(x)               # [B, 1024, 256]
#         x = self.relu(self.conv2(x))   # [B, 1024, 256]
#         x = self.pool(x)               # [B, 1024, 128]
#         x = self.relu(self.conv3(x))   # [B, 1024, 128]
#         x = self.relu(self.conv4(x))   # [B, 1024, 128]
#         x = self.relu(self.conv5(x))   # [B, 1024, 128]
#         x = self.pool(x)               # [B, 1024, 64]
#
#         x = x.flatten(start_dim=1)     # [B, 1024*64]
#         x = self.relu(self.fc1(x))     # [B, 1024]
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))     # [B, 512]
#         x = self.dropout(x)
#         x = self.fc3(x)                # [B, 512]
#
#         return x



# #
# ## add batch normalizaton and Resdual(5CNN+3FC+BN+ResNet)
# import torch
# import torch.nn as nn
# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#
#         # 基础卷积块（5层）
#         self.conv1 = nn.Conv1d(512, hidden_size, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#
#         self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#
#         self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(hidden_size)
#
#         self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm1d(hidden_size)
#
#         self.conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm1d(hidden_size)
#
#         self.pool = nn.MaxPool1d(kernel_size=2)
#
#         self.flatten_size = hidden_size * (input_size // (2 ** 3))  # = 1024 * 64
#
#         # 全连接层
#         self.fc1 = nn.Linear(self.flatten_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, output_size)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # 输入形状: [batch, 512, 512]
#         x = x.permute(0, 2, 1)  # [batch, 512, 512]
#
#         # Conv Block 1 + Pool
#         x = self.relu(self.bn1(self.conv1(x)))   # [B, 1024, 512]
#         x = self.pool(x)                         # [B, 1024, 256]
#
#         # Conv Block 2 + Pool
#         residual = x                             # 残差
#         x = self.relu(self.bn2(self.conv2(x)))   # [B, 1024, 256]
#         x = self.pool(x)                         # [B, 1024, 128]
#         x = x + self.pool(residual)              # 残差连接
#
#         # Conv Block 3 + 4 + 5 (不再下采样)
#         res = x
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = x + res  # 残差连接
#
#         x = self.pool(x)  # 再次下采样: [B, 1024, 64]
#
#         # Flatten + FC
#         x = x.flatten(start_dim=1)       # [B, 1024 * 64]
#         x = self.relu(self.fc1(x))       # [B, 1024]
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))       # [B, 512]
#         x = self.dropout(x)
#         x = self.fc3(x)                  # [B, output_size]
#
#         return x


##### Transform  ####
import torch
import torch.nn as nn

class SmilesModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024, output_size=512,
                 transformer_embed_dim=512, num_heads=8, num_layers=2):
        super(SmilesModel, self).__init__()

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FC layers
        self.fc1 = nn.Linear(input_size * transformer_embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: [batch, 512, 512]
        # 每一行为一个token，每个token是512维，符合 transformer 输入格式
        x = self.transformer(x)                  # -> [batch, 512, 512]

        x = x.view(x.size(0), -1)                # Flatten to [batch, 512*512]
        x = self.relu(self.fc1(x))               # -> [batch, hidden_size]
        x = self.dropout(x)
        x = self.fc2(x)                          # -> [batch, output_size]

        return x


#orign

# class SmilesModel(nn.Module):
#     def __init__(self, input_size=512, hidden_size=1024, output_size=512):
#         super(SmilesModel, self).__init__()
#         # 1D卷积处理局部特征
#         self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#
#         # 全连接层
#         self.fc1 = nn.Linear(hidden_size * (input_size // 4), hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         # 激活函数和正则化
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x形状: [batch, max_length, max_length]
#
#         # 首先对每行做1D卷积
#         x = x.permute(0, 2, 1)  # 变为[batch, max_length, max_length]
#
#         # 第一卷积层
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#
#         # 第二卷积层
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#
#         # 展平
#         x = x.flatten(start_dim=1)
#
#         # 全连接层
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x


