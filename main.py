import torch
import torch.optim as optim
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from timm.models import create_model

import numpy as np
import os
import argparse

import matplotlib.pyplot as plt

import utils
import pruner
import models

parser = argparse.ArgumentParser()

# model config
# 'mixer_b16_224_in21k', 'mixer_l16_224_in21k', 'vit_base_patch16_224_dino', 'vit_base_patch16_224_sam'
parser.add_argument('--model', default='vit_base_patch16_224_in21k')
parser.add_argument('--pretrain', action='store_true')

# training config
parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tiny', 'imagenet'], default='cifar10')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=150)

# pruning config
parser.add_argument('--pruning', action='store_true')
parser.add_argument('--sparsity', type=float, default=0.98)
parser.add_argument('--method', choices=['random', 'magnitude', 'snip', 'grasp', 'synf', 'snip_magnitude', 'ReFer'], default='ReFer')
parser.add_argument('--alpha', type=float)

# else
parser.add_argument('--save_dir', default='/mnt/outputs')
parser.add_argument('--seed', type=int, default=428)

args = parser.parse_args()

def main():
    device = torch.cuda.device_count()
    print('num gpus:', device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tiny':
        num_classes = 200
    elif args.dataset == 'imagenet':
        num_classes = 1000
    utils.mixup_args['num_classes'] = num_classes
    train_loader, valid_loader, half_train_loader, half_valid_loader = utils.get_dataloader(args.dataset, args.batch_size, data_download=True)
    model = create_model(args.model, pretrained=args.pretrain)

    if args.pretrain:
        init_data = model.state_dict()
    
    # pruning
    if args.pruning:
        print('pruning method:', args.method)
        print('sparsity:', args.sparsity)

        if args.method == 'synf':
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].abs()
            model.load_state_dict(state_dict)

        if 'mixer' in args.model:
            rm_modules = models.mixer_rm_modules(model)
        elif 'vit' in args.model:
            rm_modules = models.vit_rm_modules(model)
        elif 'pool' in args.model:
            rm_modules = models.pool_rm_modules(model)
        elif 'resnet' in args.model:
            rm_modules = models.resnet_rm_modules(model)
        elif 'vgg' in args.model:
            rm_modules = models.vgg_rm_modules(model)
            
        if 'dino' in args.model:
            head = nn.Linear(in_features=model.embed_dim, out_features=num_classes)
            model = nn.Sequential(model, head)
        elif 'resnet' in args.model:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in args.model:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)
        
        if args.method == 'random':
            pruner.SCORE = pruner.random(rm_modules)
        elif args.method == 'magnitude':
            pruner.SCORE = pruner.magnitude(rm_modules)
        elif args.method == 'snip':
            pruner.SCORE = pruner.snip(model, rm_modules, train_loader, device)
        elif args.method == 'synf':
            pruner.SCORE = pruner.synflow(model, rm_modules, train_loader, device)
        elif args.method == 'grasp':
            pruner.SCORE = pruner.grasp(model, rm_modules, train_loader, device)
        elif args.method == 'snip_magnitude':
            pruner.SCORE = pruner.snip_magnitude(model, rm_modules, train_loader, device, args.alpha)
        elif args.method == 'ReFer':
            pruner.SCORE = pruner.ReFer(model, rm_modules, train_loader, device)
        
        model = create_model(args.model, pretrained=args.pretrain)
        if args.pretrain:
           model.load_state_dict(init_data)
        
        if 'mixer' in args.model:
            rm_modules = models.mixer_rm_modules(model)
        elif 'vit' in args.model:
            rm_modules = models.vit_rm_modules(model)
        elif 'pool' in args.model:
            rm_modules = models.pool_rm_modules(model)
        elif 'resnet' in args.model:
            rm_modules = models.resnet_rm_modules(model)
        elif 'vgg' in args.model:
            rm_modules = models.vgg_rm_modules(model)

        if 'dino' in args.model:
            head = nn.Linear(in_features=model.embed_dim, out_features=num_classes)
            model = nn.Sequential(model, head)
        elif 'resnet' in args.model:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in args.model:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)

        pruner.prune.global_unstructured(
            rm_modules,
            pruning_method=pruner.Pruner,
            amount = args.sparsity
        )
    
    else: #Pruningしない場合も分類層の置き換えを行う
        if 'dino' in args.model:
            head = nn.Linear(in_features=model.embed_dim, out_features=num_classes)
            model = nn.Sequential(model, head)
        elif 'resnet' in args.model:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in args.model:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)

    # saving config
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_path = args.save_dir + r'/' + args.model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if args.pruning:
        save_path = save_path + r'/' + args.method
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if args.method == 'snip_magnitude':
        # file_name = 'alpha{}_sparsity{}.chkpt'.format(args.alpha, int(args.sparsity * 100))
        file_name = '{}_alpha{}_sparsity{}.chkpt'.format(args.dataset, args.alpha, int(args.sparsity * 100))
    else:
        if not args.pruning:
            # file_name = 'non_pruning.chkpt'
            file_name = '{}_non_pruning.chkpt'.format(args.dataset)
        else:
            # file_name = 'sparsity{}.chkpt'.format(int(args.sparsity * 100))
            file_name = '{}_sparsity{}.chkpt'.format(args.dataset, int(args.sparsity * 100))
            if not args.pretrain:
                file_name = 'fs' + file_name


    history = {
        'train_loss': [],
        'valid_top1_acc': [],
        'valid_top5_acc': [],
        'valid_loss': [],
        'best_epoch': 0,
        'best_model': None
    }
    # training
    best_loss = float('inf')
    device = [i for i in range(device)]
    
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')
    optimizer = optim.Adam(model.module.parameters(), lr=0.001)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, 
        lr_min=1e-4, 
        warmup_t=10, 
        warmup_lr_init=5e-5, 
        warmup_prefix=True
        )
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        scheduler.step(epoch=epoch)
    
        train_loss = utils.train_multi_gpu(
            model, train_loader, loss_func, optimizer, epoch+1, mixup=True
        )
        valid_loss, valid_top1_acc, valid_top5_acc = utils.validation_multi_gpu(
            model, valid_loader, loss_func, device, epoch+1
        )
        model.zero_grad()
        
        print('top1-acc:', valid_top1_acc)
        if epoch == 0:
            best_loss = valid_loss
            history['best_epoch'] = epoch
            history['best_model'] = model.state_dict()
            print('updated best accuracy!')
        else:
            if(valid_top1_acc > max(history['valid_top1_acc'])):
                best_loss = valid_loss
                history['best_epoch'] = epoch
                history['best_model'] = model.state_dict()
                print('updated best accuracy!')   

        history['train_loss'].append(train_loss)   
        history['valid_top1_acc'].append(valid_top1_acc)
        history['valid_top5_acc'].append(valid_top5_acc)
        history['valid_loss'].append(valid_loss)
        
        print('best acc:', max(history['valid_top1_acc']))
        
        torch.save(history, os.path.join(save_path, file_name))
               
    print(len(history['train_loss']))
    x = np.arange(0,epoch+1)
    plt.plot(x,history['valid_top1_acc'])
    plt.plot(history['best_epoch'],history['valid_top1_acc'][history['best_epoch']],marker='.')
    plt.text(history['best_epoch'],history['valid_top1_acc'][history['best_epoch']]-0.05, str(history['valid_top1_acc'][history['best_epoch']]),ha='center',va='bottom')
    plt.xlabel("epoch")
    plt.ylabel("valid_top1_acc")
    plt.title("Method:%s, sparsity:%s,   best-epoch:%s"%(args.method,str(args.sparsity),str(history['best_epoch'])))
    plt.xlim(0,epoch)
    plt.ylim(0,1)
    plt.savefig(save_path+"/Accuracy.png")
    plt.savefig(save_path+"/Accuracy.jpg")
    plt.savefig(save_path+"/Accuracy.svg")
    plt.show()
    np.save(
    save_path+"/Accuracy.npy", # データを保存するファイル名
    history["valid_top1_acc"],  # 配列型オブジェクト（listやnp.array)
    )
               
if __name__ == '__main__':
    main()
