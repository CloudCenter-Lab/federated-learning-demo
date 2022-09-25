import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import syft as sy

# 对pytorch库进行扩展
hook = sy.TorchHook(torch)
# 定义客户端
Client_1 = sy.VirtualWorker(hook,id="Client_1")
Client_2 = sy.VirtualWorker(hook,id="Client_2")
# 定义训练参数
class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 50
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

args = Arguments()
# use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cpu")
kwargs = {}

# 将训练数据集转化为联邦学习训练数据集
federated_train_loader = sy.FederatedDataLoader(datasets.MNIST('../data', train = True, download= True, transform = transforms.Compose([ transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ])).federate((Client_1,Client_2)), batch_size=args.batch_size, shuffle=True, **kwargs)
# 测试数据集保持不变
test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
batch_size = args.test_batch_size, shuffle = True, **kwargs)

# 定义cnn网络（模型不一样，则替换此处代码）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4*4*50,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

# 定义训练函数
def train(args, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        #发送至客户端
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        #获得客户端模型
        model.get()
        if batch_idx % args.log_interval==0:
        #获得loss
            loss = loss.get()
            print('Train Epoch : {} [ {} / {} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) *
                    args.batch_size,
                100.* batch_idx / len(federated_train_loader), loss.item()))

# 定义测试函数
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            dataset, target = data.to(device), target.to(device)
            output=model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)                # 损失函数总和
            correct+= pred.eq(target.view_as(pred)).sum().item() # 获取最大对数概率的索引
        test_loss/= len(test_loader.dataset)
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.* correct / len(test_loader.dataset)))

# 开始训练
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs +1):
    train(args, model, device, federated_train_loader, optimizer, epoch)
    test(args, model, device, test_loader)

#保存模型
if (args.save_model):
    torch.save_modelave(model.state_dict(),"mnist_cnn.pt")