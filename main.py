import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
from model import ResNet
import argparse
import os


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=30, type=int, help='epoch to train your network') 
parser.add_argument('--batchsize', default=16, type=int, help='batchsize') 
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', default=None, type=str, help='load checkpoint')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay of params in optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--test_iter', default=3, type=int, help='every n epochs to evaluate') 
parser.add_argument('--outdir', default='outdir', help='folder to output images and model checkpoints')
parser.add_argument('--result_dir', default='test_acc_log', help='folder to save test result log') 
args = parser.parse_args()

# Get data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,
                                       download=False, transform=transform)
#定义测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet().to(device)

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if __name__ == "__main__":
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    # create output dir
    dir_name = os.path.join(args.outdir,'result_lr{}_bs{}'.format(args.lr,args.batchsize))
    os.makedirs(dir_name,exist_ok=True)

    print("Start training!")
    evaluate_dict = {}
    for epoch in range(args.epochs):
        print('\nEpoch: %d' % (epoch))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for idx, data in enumerate(trainloader):
            batch_num = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            if(idx % 100 == 0):
                cur_loss = sum_loss / (idx + 1)
                cur_acc = correct / total * 100
                print('[epoch:{} iter:{}]: training_acc = {}% , current loss: {}'.format(epoch, idx, cur_acc, cur_loss))

        # evaluate on test set
        if ((epoch+1) % args.test_iter == 0):
            print("epoch:{} , now testing!".format(epoch+1))
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                test_acc = torch.true_divide(100*correct, total)
                test_acc = float(test_acc.cpu().numpy())
                print('test acc：%.3f%%' % test_acc)
                evaluate_dict[epoch+1] = test_acc

                # Save models
                # file_name = os.path.join(dir_name,'epoch_{}.pth'.format(epoch+1))
                # torch.save(net.state_dict(),file_name)

    # save test_acc file
    dir = args.result_dir
    os.makedirs(dir,exist_ok=True)
    file_name = os.path.join(dir,'lr{}_bs{}log.json'.format(args.lr,args.batchsize))
    with open(file_name, 'w') as f:
        json.dump(evaluate_dict, f)
