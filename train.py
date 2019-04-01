import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import MyDataset
import argparse
from network import resnet18


def adjust_learning_rate(optimizer, decay_rate=.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def trainandsave(args):

    loss_log = open("loss_log.txt",'w')

    trainData = MyDataset('train.txt',train=True)
    testData = MyDataset('val.txt',train=False)
    train_loader = DataLoader(trainData,batch_size=args.batch,num_workers=4,shuffle=True)
    test_loader = DataLoader(testData,batch_size=32,num_workers=4,shuffle=False,drop_last=True)
    net = resnet18(pretrained=True, num_classes=5,drop_out=args.drop_out)
    if(args.gpu):
        net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        net.train()
        if(epoch in args.lr_decay_steps):
            adjust_learning_rate(optimizer,0.1)
        running_loss = 0.0
        for batch, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            labels = labels.squeeze(1)
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            if(args.gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().data
            loss_log.write(str(float(loss.cpu().data))+' ')
            if batch % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch, batch, running_loss / 10))
                running_loss = 0.0
        #TEST
        net.eval()
        total = 0
        correct = 0
        for batch, data in enumerate(test_loader):
            inputs, labels = data
            labels = labels.squeeze(1)
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            if (args.gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            for idx in range(outputs.size(0)):
                pred = torch.argmax(outputs[idx])
                gt = labels[idx]
                if(int(pred.cpu().data) == int(gt.cpu().data)):
                    correct+=1
                total  += 1
        torch.save(net, 'models/net_'+str(epoch)+'.pkl')
        print "total", total, "acc ",float(correct)/total
    loss_log.close()
    print('Finished Training')

    # torch.save(net.state_dict(), 'net_params.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch",type=int,default="64")
    parser.add_argument("-data_path",type=str,default='/home/zhou/mineral_data/trainval')
    parser.add_argument("-gpu",type=bool,default=True)
    parser.add_argument("-lr", type=float, default='0.001')
    parser.add_argument("-momentum",type=float,default='0.9')
    parser.add_argument("-epochs", type=int, default='5')
    parser.add_argument("-lr_decay_steps",type=list,default=[1,3])
    parser.add_argument("-wd", type=float, default='5e-4')
    parser.add_argument("-drop_out",type= float,default=0.5)
    args = parser.parse_args()
    trainandsave(args)