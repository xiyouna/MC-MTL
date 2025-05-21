##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from pre_net import PreNet


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class ClassifierHead(nn.Module):
    def __init__(self, num_features, num_cls):
        super(ClassifierHead, self).__init__()
        # 添加平均池化层，注意这里我们使用全局平均池化作为示例
        # 如果你需要其他类型的池化，可以替换为nn.AvgPool2d或nn.MaxPool2d等，并设置适当的kernel_size和stride
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        # 将池化后的特征展平
        self.flatten = nn.Flatten()
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1000),  # 注意这里的num_features需要根据特征图的通道数C来确定
            nn.ReLU(),
            nn.Linear(1000, num_cls)
        )

    def forward(self, x):
        # 通过池化层
        x = self.pool(x)
        # 展平池化后的特征
        x_flatten = self.flatten(x)
        # 通过分类器
        x = self.classifier(x_flatten)
        return x, x_flatten

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        # z_dim = 512
        z_dim = 640
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            # self.encoder = ResNetMtl()
            self.encoder = PreNet(out_channels=640, mtl=True)
            # self.net = CAM()
        else:
            self.encoder = PreNet(args)
            self.pre_fc = ClassifierHead(num_features=640, num_cls=num_cls)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """

        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        inp_f = self.encoder(inp)
        # print(inp_f.shape)torch.Size([1, 512, 5, 5])
        x, x_flatten = self.pre_fc(inp_f)
        return x, inp_f

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """

        batch_size, num_train = data_shot.size(0), data_shot.size(1)
        num_test = data_query.size(1)
        K = label_shot.size(2)
        label_shot = label_shot.transpose(1, 2)

        data_shot = data_shot.view(-1, data_shot.size(2), data_shot.size(3), data_shot.size(4))
        data_query = data_query.view(-1, data_query.size(2), data_query.size(3), data_query.size(4))

        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)

        ytest, cls_scores = self.net(images_train, images_test)
        logits = self.base_learner(embedding_shot)

        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        # print("embedding_query1 shape:", embedding_query.shape)  # (batch_size, num_classes)

        embedding_shot = self.encoder(data_shot)
        x, embedding_shot = self.pre_fc(embedding_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        x, embedding_query = self.pre_fc(embedding_query)
        # print("embedding_query shape:", embedding_query.shape)  # (batch_size, num_classes)

        logits_q = self.base_learner(embedding_query, fast_weights)
        # print("logits_q shape:", logits_q.shape)  # (batch_size, num_classes)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q
        