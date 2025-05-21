import os
import pickle

import numpy as np
import tqdm
import time
# import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run, cp_accuracy
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.model import McMtl
from test import test_main, evaluate
from sklearn.linear_model import LogisticRegression


def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query - base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov


def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()

    loss_meter = Meter()
    acc_meter = Meter()
    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = "MC-MTL/checkpoints/DC_pre/cifar_fs_test1_features.plk"
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()
        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'tc'
        data_shot, data_query = data[:k], data[k:]
        absolute_logits, _, qry_dc, spt_dc = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))

        absolute_loss = F.cross_entropy(absolute_logits, label)
        acc = cp_accuracy(absolute_logits, label)

        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)

        loss = args.lamb * absolute_loss + loss_aux

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    if torch.cuda.is_available():
        print("GPU is available. Using GPU...")
    else:
        print("GPU is not available. Using CPU but this is not recommended.")
        raise EnvironmentError("GPU is not available. This project requires a GPU to run.")

    set_seed(args.seed)
    model = McMtl(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    print("==> Load pre_net")
    pretrained_path = f'MC-MTL/checkpoints/pre/{args.pre_pth}.pth'
    pretrained_dict = torch.load(pretrained_path)['params']
    loaded_0 = set(pretrained_dict.keys())

    prenet_dict = {k.replace('resnet.', ''): v for k, v in pretrained_dict.items() if k.startswith('resnet.')}

    prenet_dict = {k: v for k, v in prenet_dict.items() if k in model.module.encoder.state_dict()}
    model.module.encoder.load_state_dict(prenet_dict, strict=False)
    loaded_keys = set(prenet_dict.keys()).intersection(set(model.module.encoder.state_dict().keys()))

    print(f"Number of loaded keys: {len(loaded_keys)}")
    print(f"Total keys in model encoder: {len(model.module.encoder.state_dict())}")

    prenet_dict = {k.replace('sc_module.', ''): v for k, v in pretrained_dict.items() if k.startswith('sc_module.')}
    prenet_dict = {k: v for k, v in prenet_dict.items() if k in model.module.scr_module.state_dict()}
    model.module.scr_module.load_state_dict(prenet_dict, strict=False)
    loaded_keys = set(prenet_dict.keys()).intersection(set(model.module.scr_module.state_dict().keys()))
    print(f"Number of loaded keys: {len(loaded_keys)}")
    print(f"Total keys in model encoder: {len(model.module.scr_module.state_dict())}")

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)
    print("==> train")
    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        if val_acc > max_acc:
            print(f'[ log ] -----------A better model is found ({val_acc:.3f}) -----------')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)
