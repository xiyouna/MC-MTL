##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import collections
import os.path as osp
import os
import pickle

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset

class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        torch.autograd.set_detect_anomaly(True)

        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, str(args.shot)])
        # save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
        #     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        # args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        args.save_path = pre_base_dir + '/' + save_path1

        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        # self.trainset = Dataset('train', self.args, train_aug=True)
        self.trainset = Dataset('test', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 300, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class
        
        # Build pretrain model
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)

        # 加载预训练
        # 为了保存训练程度高的特征分布数据
        self.model_dict = self.model.state_dict()
        pretrained_dict = torch.load('/media/breeze/zxn/meta-transfer-learning-scr/pytorch/logs/pre/cub_5/max_acc.pth')['params']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)

        # Set optimizer 编码器和预训练全连接层参数分别进行优化 但学习率相同
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
            {'params': self.model.pre_fc.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        # 只保存编码器参数不保存分类器
        torch.save(dict(params=self.model.module.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    # 保存pickle文件，file为路径
    def save_pickle(self, file, data):
        with open(file, 'wb') as f:
            pickle.dump(data, f)  # 序列化data写入f

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        
        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):

            # Set the model to train mode
            self.model.train()
            self.model.module.mode = 'pre'
            # for name, module in self.model.named_modules():
            #     print(name)
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            output_dict = collections.defaultdict(list)
            save_dir = '/media/breeze/zxn/meta-transfer-learning-scr/pytorch/logs'  # 检查点目录
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                labels = label
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                # Output logits for model
                # print(data.shape)torch.Size([1, 3, 80, 80])
                logits, inp_f = self.model(data)

                # 存储特征
                outputs = inp_f.mean(dim=[-1, -2])
                outputs = outputs.cpu().data.numpy()  # 转化为numpy数组
                # 将输出添加到与标签对应的列表中
                for out, label_save in zip(outputs, labels):
                    output_dict[label_save.item()].append(out)


                # Calculate train loss
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 保存特征
            all_info = output_dict
            self.save_pickle(save_dir + '/cub_test1_features.plk', all_info)
            # Update learning rate
            self.lr_scheduler.step()
            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.module.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test 
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            # Generate the labels for train
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)
              
            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                # print("p:", p)  # (batch_size,)
                # print("data:", data.shape)  # (batch_size, num_classes)

                data_shot, data_query = data[:p], data[p:]
                # print("Logits shape:", data_query.shape)  # (batch_size, num_classes)

                logits = self.model((data_shot, label_shot, data_query))
                # print("Logits shape:", logits.shape)  # (batch_size, num_classes)
                # print("Label shape:", label.shape)  # (batch_size,)

                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        writer.close()
        