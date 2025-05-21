import os
import pickle

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.model import McMtl


def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query - base_means[i]))
    index = np.argpartition(dist, k)[:k]

    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    # ---- Base class statistics
    base_means = []
    base_cov = []
    num = 0
    base_features_path = "MC-MTL/checkpoints/DC_pre/mini_testset_features.plk"
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            # 对特征算均值和协方差
            # print(feature.shape)
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
            num += 1
    print('存储特征类别数', num)
    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            labels = labels.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            qry_num = data_query.size(0)

            model.module.mode = 'tc'
            _, logits, _, _ = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))

            # 初始支持集\查询集
            spt_original = data_shot.mean(dim=[-1, -2])
            qry_dc = data_query.mean(dim=[-1, -2])
            qry_dc = qry_dc.cpu().detach().numpy()
            spt_original = spt_original.cpu().detach().numpy()

            # ---- Tukey's transform
            beta = 1
            support_data = np.power(spt_original[:, ], beta)
            query_data = np.power(qry_dc[:, ], beta)

            # ---- distribution calibration and feature sampling
            support_label = torch.arange(args.way).repeat(args.shot).detach().cpu().numpy()
            query_label = torch.arange(args.way).repeat(args.query).detach().cpu().numpy()

            sampled_data = []
            sampled_label = []
            num_sampled = int(750 / args.shot)
            n_lsamples = data_shot.size(0)
            for i in range(n_lsamples):
                spt_dc = support_data[i]
                mean, cov = distribution_calibration(spt_dc, base_means, base_cov, k=1)
                sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
                sampled_label.extend([support_label[i]] * num_sampled)
            sampled_data = np.concatenate([sampled_data[:]]).reshape(args.way * args.shot * num_sampled, -1)
            X_aug = np.concatenate([support_data, sampled_data])
            Y_aug = np.concatenate([support_label, sampled_label])

            # ---- train classifier，简单逻辑回归分类/可考虑K近邻
            assert not np.isnan(X_aug).any(), "X_aug contains NaN values"
            classifier = LogisticRegression(max_iter=800).fit(X=X_aug, y=Y_aug)

            predicts_dc = classifier.predict(query_data)
            loss = F.cross_entropy(logits, labels[k:])
            acc = compute_accuracy(predicts_dc, query_label, qry_num)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = McMtl(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
