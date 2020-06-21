'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

if __name__=='__main__':


    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torch.backends.cudnn as cudnn
    from torch.optim.lr_scheduler import MultiStepLR
    from dataset import AIGS10

    import torchvision
    import torchvision.transforms as transforms

    import os
    import argparse

    from models.resnet import *
    from utils import progress_bar
    from torch.autograd import Variable

    optimizers_dict={'SGD':optim.SGD,'Adam':optim.Adam}


    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--data',default='cifar10',choices=['cifar10','AIGS10'])
    parser.add_argument('--rotate',action='store_true')

    parser.add_argument('--opt',default='SGD',choices=['SGD','Adam'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--n_epochs', default=350, type=int)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint',default='ckpt')

    parser.add_argument('--attention',action='store_true')
    parser.add_argument('--num_f1',type=int,default=23)
    parser.add_argument('--num_f2',type=int,default=45)
    parser.add_argument('--num_f3',type=int,default=91)
    parser.add_argument('--num_f4',type=int,default=181)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    n_epochs = args.n_epochs
    lr=args.lr

    # Data
    means = (0.4914, 0.4822, 0.4465)
    print('==> Preparing data..')

    transform_train=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])
    if args.rotate:
        print('Randomly Rotate & Flip images...')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])

    if args.data=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        trainset = AIGS10(train=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = AIGS10(train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        classes = ['aeroplane', 'car', 'bird', 'cat', 'sheep', 'dog', 'chair', 'horse', 'boat', 'train']


    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'{args.checkpoint}.t7'))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        lr=checkpoint['lr']
    else:
        print('==> Building model..')
        net = ResNet18(args.num_f1,
                       args.num_f2,
                       args.num_f3,
                       args.num_f4,
                       args.attention)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if args.opt=='SGD':
        optimizer = optimizers_dict[args.opt](net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optimizers_dict[args.opt](net.parameters(),lr=lr)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        global best_acc
        global optimizer
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:
                for param_group in optimizer.param_groups:
                    lr = float(param_group['lr'])
                print('Saving..')
                state = {
                    'net': net.module if use_cuda else net,
                    'acc': acc,
                    'epoch': epoch,
                    'lr':lr
                }
                if not os.path.isdir(args.checkpoint_dir):
                    os.mkdir(args.checkpoint_dir)
                torch.save(state, os.path.join(args.checkpoint_dir, f'{args.checkpoint}.t7'))
                best_acc = acc

    milestones = [50, 100, 140]
    scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)

    if args.resume:
        start_epoch=start_epoch+1
        n_epochs=start_epoch+n_epochs

    for epoch in range(start_epoch, n_epochs):
        train(epoch)
        scheduler.step()
        test(epoch)

