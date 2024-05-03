import torch
import torch.nn as nn

bs = 3  # 举例的批量大小
num_labels = 16  # 标签的数量

# 模拟模型输出，假设是原始的logits
logits = torch.randn(bs, num_labels, requires_grad=True)

# 目标张量，每个元素是0或1，表示标签是否适用于对应的样本
targets = torch.empty(bs, num_labels).random_(2)
print(logits.shape)
print(targets.shape)
print(logits.dtype)
print(targets.dtype)
print(logits)
print(targets)
# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 计算损失
loss = criterion(logits, targets)

# 反向传播
loss.backward()

# 打印损失值
print(loss.item())
