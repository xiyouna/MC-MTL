import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from sklearn.linear_model import LogisticRegression

from models.resnet import ResNet
from models.cca import CCA
from models.scr import SCR, SelfCorrelationComputation
from models.others.se import SqueezeExcitation
from models.others.lsa import LocalSelfAttention
from models.others.nlsa import NonLocalSelfAttention
from models.others.sce import SpatialContextEncoder
from models.others.trip import TripletAttention
from models.others.resnet_t import ResidualNet
from models.others.bdc_module import BDC




class RENet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        # self.encoder = ResidualNet("ImageNet", depth=18, att_type=True)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.fc_q = nn.Linear(self.encoder_dim, self.args.num_class)

        self.dcov = BDC(is_vec=True, input_dim=[640, 10, 10], dimension_reduction=640)

        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        if self.args.self_method == 'scr':
            corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            self_block = SCR(planes=planes, stride=stride)

        elif self.args.self_method == 'sce':
            planes = [640, 64, 64, 640]
            self_block = SpatialContextEncoder(planes=planes, kernel_size=kernel_size[0])
        elif self.args.self_method == 'se':
            self_block = SqueezeExcitation(channel=planes[0])
        elif self.args.self_method == 'lsa':
            self_block = LocalSelfAttention(in_channels=planes[0], out_channels=planes[0], kernel_size=kernel_size[0])
        elif self.args.self_method == 'nlsa':
            self_block = NonLocalSelfAttention(planes[0], sub_sample=False)
        elif self.args.self_method == 'trip':
            self_block = TripletAttention(640, no_spatial=False)
        else:
            raise NotImplementedError

        if self.args.self_method == 'scr':
            layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
            # return self.bdc(input)

        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        spt = spt.squeeze(0)

        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        qry_cam = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        # spt_attended_pooled = spt_attended.view(spt_attended.size(0), spt_attended.size(1), -1)
        # qry_pooled = qry.mean(dim=[-1, -2])

        qry_pooled = qry_attended_pooled.mean(dim=1)
        # print(qry.shape, qry_attended.shape)torch.Size([37, 640, 5, 5]) torch.Size([37, 5, 640, 5, 5])
        # # 无掩码权重的查询送入求相似度
        # qry = qry.mean(dim=[-1, -2])
        # # qry = qry.view(qry.size(0), -1)
        # qry = qry.view(num_qry, -1)
        # m = spt_attended_pooled.size(1)
        # n = qry.size(0)
        # d = qry.size(1)
        # qry_pro = qry.unsqueeze(1).expand(n, m, d)

        # print(spt_attended_pooled.shape, qry_pro.shape, qry.shape)torch.Size([37, 5, 640]) torch.Size([37, 5, 16000]) torch.Size([37, 16000])
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        qry_dc = qry_pooled

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc_q(qry_pooled), qry_dc, spt, qry_cam
            # return self.fc_q(qry_pooled), qry_dc, spt
        else:
            return similarity_matrix / self.args.temperature, self.fc_q(qry_pooled), qry_dc, spt, qry_cam
            # return self.fc_q(qry_pooled), qry_dc, spt

    def bdc(self, x):
        # print(spt.shape,qry.shape)25, 640, 5, 5[38, 640, 5, 5]
        x = x.squeeze(0)
        # print(x.shape)[100, 640, 5, 5]
        # x = x.contiguous().view(self.args.way * (self.args.shot + self.args.query), *x.size()[1:])
        z_all = self.dcov(x)
        # z_all = z_all.view(self.args.way, self.args.shot + self.args.query, -1)
        # spt = z_all[:, :self.args.shot]
        # qry = z_all[:, self.args.shot:]
        #
        # z_proto = spt.contiguous().view(self.args.way, self.args.shot, -1).mean(1)
        # z_query = qry.contiguous().view(self.args.way*self.args.query, -1)
        # scores = self.metric(z_query, z_proto)

        return z_all


    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.args.shot > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        '''
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        '''
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x
            x = self.scr_module(x)

            if self.args.self_method == 'scr':
                x = x + identity
            x = F.relu(x, inplace=True)

        # self.args.self_method = 'sce'
        # x = self.scr_module(x)
        # x = F.relu(x, inplace=True)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x
