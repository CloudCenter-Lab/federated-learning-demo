import torch
import syft as sy
import copy
from torch import nn,optim

hook = sy.TorchHook(torch)

# 创建工作机
Han = sy.VirtualWorker(hook,id="Han")
Zhang = sy.VirtualWorker(hook,id="Zhang")
secure_worker = sy.VirtualWorker(hook,id="secure_worker")

# 创建数据
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]],requires_grad=True)
target = torch.tensor([[0],[0],[1],[1.]],requires_grad=True)

# 将训练数据发送给工作机，数据一共分为两份
data_Han = data[0:2]
target_Han = target[0:2]
data_Zhang = data[2:]
target_Zhang = target[2:]
data_Han = data_Han.send(Han)
target_Han = target_Han.send(Han)
data_Zhang = data_Zhang.send(Zhang)
target_Zhang = target_Zhang.send(Zhang)

# 创建模型
model = nn.Linear(2,1)

# 训练模型
iterations = 20
worker_iters = 5
for a_iter in range(iterations):
    #发送模型给工作机
    Han_model = model.copy().send(Han)
    Zhang_model = model.copy().send(Zhang)
    # 定义工作机的优化器
    Han_opt = optim.SGD(params=Han_model.parameters(),lr=0.2)
    Zhang_opt = optim.SGD(params=Zhang_model.parameters(),lr=0.1)
    # 开始训练
    for wi in range(worker_iters):
        Han_opt.zero_grad()
        # 训练Han的模型
        Han_pred = Han_model(data_Han)
        Han_Loss = ((Han_pred-target_Han)**2).sum()
        Han_Loss.backward()
        Han_opt.step()
        Han_Loss = Han_Loss.get().data
        # 训练Zhang的模型
        Zhang_pred = Zhang_model(data_Zhang)
        Zhang_Loss = ((Zhang_pred - target_Zhang) ** 2).sum()
        Zhang_Loss.backward()
        Zhang_opt.step()
        Zhang_Loss = Zhang_Loss.get().data

    # 将更新后的模型发送给服务器进行聚合
    Zhang_model.move(secure_worker)
    Han_model.move(secure_worker)
    # 模型平均
    with torch.no_grad():
        model.weight.set_(((Zhang_model.weight.data + Han_model.weight.data) / 2).get())
        model.bias.set_(((Zhang_model.bias.data + Han_model.bias.data) / 2).get())
    print("Han" + str(Han_Loss) + "Zhang" + str(Zhang_Loss))

# 模型评估
pred = model(data)
loss = ((pred - target)**2).sum()
print("预测值:",pred)
print("目标值:",target)
print("损失:",loss.data)

