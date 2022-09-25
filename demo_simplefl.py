import torch
from torch import nn
from torch import optim
import syft as sy

hook = sy.TorchHook(torch)

# 创建工作机
Han = sy.VirtualWorker(hook,id="Han")
Zhang = sy.VirtualWorker(hook,id="Zhang")

# 创建数据和模型
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]],requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]],requires_grad=True)
model = nn.Linear(2,1)

# 将训练数据发送给工作机，数据一共分为两份
data_Han = data[0:2]
target_Han = target[0:2]
data_Zhang = data[2:]
target_Zhang = target[2:]

# 将训练数据发送给工作机
data_Han = data_Han.send(Han)
target_Han = target_Han.send(Han)
data_Zhang = data_Zhang.send(Zhang)
target_Zhang = target_Zhang.send(Zhang)

# 存储张量指针
datasets = [(data_Han,target_Han),(data_Zhang,target_Zhang)]

def train():
    opt = optim.SGD(params=model.parameters(),lr=0.1)
    for iter in range(50):
        # 遍历每个工作机的数据集
        for data,target in datasets:
            # 将模型发送给对应的工作机
            model.send(data.location)
            # 消除之前的梯度
            opt.zero_grad()
            # 预测
            pred = model(data)
            # 计算损失
            loss = ((pred - target)**2).sum()
            # 回传损失
            loss.backward()
            # 更新参数
            opt.step()
            # 获取模型
            model.get()

            print(loss.data)


train()


