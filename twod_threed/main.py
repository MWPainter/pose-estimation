#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
import time
from pprint import pprint
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data.distributed
import horovod.torch as hvd

from twod_threed.src.procrustes import get_transformation
import twod_threed.src.data_process as data_process
from twod_threed.src import Bar
import twod_threed.src.utils as utils
import twod_threed.src.misc as misc
import twod_threed.src.log as log

from twod_threed.src.model import LinearModel, weight_init, Discriminator, ProjectNet
from twod_threed.src.datasets.human36m import Human36M

from utils import data_utils
from utils.plotting_utils import *
from utils.osutils import mkdir_p, isdir

from tensorboardX import SummaryWriter

from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio


def main(opt):
    if opt.cycle_gan:
        _main_cycle_gan(opt)
    else:
        _main_regression(opt)



def _main_cycle_gan(opt):
    """
    Currently unused, but, couldn't face deleting
    The training loop logic for using a Cycle GAN.
    """
    pass
    # # Global variables
    # start_epoch = 0
    # err_best = 1000
    # glob_gen_step = 0
    # glob_discr_step = 0
    # lr_now = opt.lr
    #
    # # save options + make a summary writer for tensorboard
    # log.save_options(opt, opt.checkpoint_dir)
    # writer = SummaryWriter(log_dir="%s/2d3d_h36m_%s_tb_log" % (opt.tb_dir, opt.exp))
    #
    # # list of action(s)
    # actions = misc.define_actions(opt.action)
    # num_actions = len(actions)
    # print(">>> actions to use (total: {}):".format(num_actions))
    # pprint(actions, indent=4)
    # print(">>>")
    #
    # # data loading
    # print(">>> loading data")
    # # load dadasets for training
    # train_dataset, train_loader, test_loader = _make_torch_data_loaders(opt, actions)
    # print(">>> data loaded !")
    #
    # # Make normalizing and denormalizing functions
    # stat_2d = train_dataset.get_stat_2d()
    # stat_3d = train_dataset.get_stat_3d()
    #
    # unorm2d = lambda x: data_process.unNormalizeDataTorch(x, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
    # unorm3d = lambda x: data_process.unNormalizeDataTorch(x, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
    # renorm2d = lambda x: data_process.reNormalizeDataTorch(x, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
    # renorm3d = lambda x: data_process.reNormalizeDataTorch(x, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
    #
    # # Create models and setup horovod
    # print(">>> creating models")
    # G = LinearModel()
    # if not opt.use_fc_for_projection:
    #     F = ProjectNet()
    # else:
    #     F = LinearModel(input_size=48, output_size=32, linear_size=512, num_stage=1)
    # D_Y = Discriminator(dimension=3)
    # D_X = Discriminator(dimension=2)
    # if opt.use_horovod:
    #     hvd.init()
    #     torch.cuda.set_device(hvd.local_rank())
    #     args.lr *= hvd.size()
    #     G.cuda()
    #     F.cuda()
    #     D_Y.cuda()
    #     D_X.cuda()
    # else:
    #     G = G.cuda()
    #     F = F.cuda()
    #     D_Y = D_Y.cuda()
    #     D_X = D_X.cuda()
    # G.apply(weight_init)
    # D_Y.apply(weight_init)
    # D_X.apply(weight_init)
    #
    # num_params = sum(p.numel() for p in G.parameters()) \
    #              + sum(p.numel() for p in F.parameters()) \
    #              + sum(p.numel() for p in D_X.parameters()) \
    #              + sum(p.numel() for p in D_Y.parameters())
    # print(">>> total params: {:.2f}M".format(num_params / 1000000.0))
    #
    # # Setup learning rates
    # lr_F = opt.project_lr if opt.project_lr != -1.0 else opt.lr
    # lr_D_X = opt.discr_2d_lr if opt.discr_2d_lr != -1.0 else opt.lr
    # lr_D_Y = opt.discr_3d_lr if opt.discr_3d_lr != -1.0 else opt.lr
    #
    # # Criterion for (regression) loss + create seperate optimizers for generative and discriminitive models
    # criterion = nn.MSELoss(size_average=True)
    # gen_params = [{'params': G.parameters()},
    #               {'params': F.parameters(), 'lr': lr_F}]
    # discr_params = [{'params': D_X.parameters(), 'lr': lr_D_X},
    #                 {'params': D_Y.parameters(), 'lr': lr_D_Y}]
    # gen_optimizer = torch.optim.Adam(gen_params, lr=opt.lr, amsgrad=opt.use_amsprop)
    # discr_optimizer = torch.optim.Adam(discr_params, lr=opt.lr, amsgrad=opt.use_amsprop)
    # if opt.use_horovod:
    #     gen_optimizer = hvd.DistributedOptimizer(gen_optimizer, named_parameters=gen_params)
    #     discr_optimizer = hvd.DistributedOptimizer(discr_optimizer, named_parameters=discr_params)
    # else:
    #     criterion = criterion.cuda()
    #
    # # load ckpt
    # if opt.load:
    #     print(">>> loading ckpt from '{}'".format(opt.load))
    #     ckpt = torch.load(opt.load)
    #     start_epoch = ckpt['epoch']
    #     err_best = ckpt['err']
    #     glob_gen_step = ckpt['gen_step']
    #     glob_discr_step = ckpt['discr_step']
    #     lr_now = ckpt['lr']
    #     G.load_state_dict(ckpt['3d_gen_state_dict'])
    #     F.load_state_dict(ckpt['2d_gen_state_dict'])
    #     D_Y.load_state_dict(ckpt['3d_discr_state_dict'])
    #     D_X.load_state_dict(ckpt['2d_discr_state_dict'])
    #     gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
    #     discr_optimizer.load_state_dict(ckpt['discr_optimizer'])
    #     print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    # if opt.resume:
    #     logger = log.Logger(os.path.join(opt.checkpoint_dir, 'log.txt'), resume=True)
    # else:
    #     logger = log.Logger(os.path.join(opt.checkpoint_dir, 'log.txt'))
    #     logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])
    #
    # cudnn.benchmark = True
    # for epoch in range(start_epoch, opt.epochs):
    #     print('==========================')
    #     print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
    #
    #     # per epoch (not test == test when not using gan)
    #     glob_gen_step, glob_discr_step, lr_now, loss_train = _train_gan(
    #         train_loader, G, F, D_X, D_Y, unorm3d, renorm2d, criterion, gen_optimizer, discr_optimizer,
    #         discr_updates_per_gen_update=opt.discr_updates_per_gen_update,
    #         lr_init=opt.lr, lr_now=lr_now, glob_gen_step=glob_gen_step, glob_discr_step=glob_discr_step,
    #         lr_decay=opt.lr_decay, gamma=opt.lr_gamma, no_grad_clipping=opt.no_grad_clipping, grad_clip=opt.grad_clip,
    #         writer=writer, tb_log_freq=opt.tb_log_freq, use_horovod=opt.use_horovod,
    #         gan_coeff=opt.gan_coeff, cycle_coeff=opt.cycle_coeff, regression_coeff=opt.regression_coeff,
    #         gradient_penalty_coeff=opt.gradient_penalty_coeff, using_projection=not opt.use_fc_for_projection)
    #     loss_test, err_test = _test(test_loader, G, criterion, stat_3d, procrustes=opt.procrustes)
    #
    #     # Update tensorboard summaries
    #     writer.add_scalars('data/loss', {'train_loss': loss_train, 'test_loss': loss_test}, epoch)
    #     writer.add_scalar('data/test_error', err_test, epoch)
    #
    #     # update log file
    #     logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
    #                   ['int', 'float', 'float', 'flaot', 'float'])
    #
    #     # save ckpt
    #     model_specific_checkpoint_dir = "%s/2d3d_h36m_%s" % (opt.checkpoint_dir, opt.exp)
    #     if not isdir(model_specific_checkpoint_dir):
    #         mkdir_p(model_specific_checkpoint_dir)
    #
    #     is_best = err_test < err_best
    #     err_best = min(err_test, err_best)
    #     if is_best:
    #         log.save_ckpt({'epoch': epoch + 1,
    #                        'lr': lr_now,
    #                        'gen_step': glob_gen_step,
    #                        'discr_step': glob_discr_step,
    #                        'err': err_best,
    #                        '3d_gen_state_dict': G.state_dict(),
    #                        '2d_gen_state_dict': F.state_dict(),
    #                        '2d_discr_state_dict': D_X.state_dict(),
    #                        '3d_discr_state_dict': D_Y.state_dict(),
    #                        'gen_optimizer': gen_optimizer.state_dict(),
    #                        'discr_optimizer': discr_optimizer.state_dict()},
    #                       ckpt_path=model_specific_checkpoint_dir,
    #                       is_best=True)
    #     else:
    #         log.save_ckpt({'epoch': epoch + 1,
    #                        'lr': lr_now,
    #                        'gen_step': glob_gen_step,
    #                        'discr_step': glob_discr_step,
    #                        'err': err_best,
    #                        '3d_gen_state_dict': G.state_dict(),
    #                        '2d_gen_state_dict': F.state_dict(),
    #                        '2d_discr_state_dict': D_X.state_dict(),
    #                        '3d_discr_state_dict': D_Y.state_dict(),
    #                        'gen_optimizer': gen_optimizer.state_dict(),
    #                        'discr_optimizer': discr_optimizer.state_dict()},
    #                       ckpt_path=model_specific_checkpoint_dir,
    #                       is_best=False)
    #
    # logger.close()
    # writer.close()

def _main_regression(opt):
    """
    Main training loop for the 3D baseline
    """
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.checkpoint_dir)

    # Make a summary writer
    writer = SummaryWriter(log_dir="%s/2d3d_h36m_%s_tb_log" % (opt.tb_dir, opt.exp))

    # create model
    print(">>> creating model")
    model = LinearModel(dataset_normalized_input=opt.dataset_normalization)
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.checkpoint_dir, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.checkpoint_dir, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")
    # data loading

    # data loading
    print(">>> loading data")
    # load dadasets for training
    train_dataset, train_loader, test_loader = _make_torch_data_loaders(opt, actions)
    stat_3d = train_dataset.get_stat_3d()
    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        # per epoch
        glob_step, lr_now, loss_train = _train(
            train_loader, model, criterion, optimizer, writer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            no_grad_clipping=opt.no_grad_clipping, grad_clip=opt.grad_clip, tb_log_freq=opt.tb_log_freq,
            use_horovod=opt.use_horovod)
        loss_test, err_test = _test(test_loader, model, criterion, opt.dataset_normalization, procrustes=opt.procrustes)

        # Update tensorboard summaries
        writer.add_scalars('data/epoch/loss', {'train_loss': loss_train, 'test_loss': loss_test}, epoch)
        writer.add_scalar('data/epoch/validation_error', err_test, epoch)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'float', 'float'])

        # save ckpt
        model_specific_checkpoint_dir = "%s/2d3d_h36m_%s" % (opt.checkpoint_dir, opt.exp)
        if not isdir(model_specific_checkpoint_dir):
            mkdir_p(model_specific_checkpoint_dir)
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=model_specific_checkpoint_dir,
                          is_best=True)
        log.save_ckpt({'epoch': epoch + 1,
                       'lr': lr_now,
                       'step': glob_step,
                       'err': err_best,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict()},
                      ckpt_path=model_specific_checkpoint_dir,
                      is_best=False)
    logger.close()
    writer.close()





def _make_torch_data_loaders(opt, actions):
    """
    Load the PyTorch datasets and data loaders.
    """
    train_dataset = Human36M(actions=actions, data_path=opt.data_dir,
                             orthogonal_data_augmentation_prob=opt.orthogonal_data_augmentation_prob,
                             z_rotations_only=opt.z_rotations_only, dataset_normalization=opt.dataset_normalization,
                             flip_prob=opt.flip_prob, drop_joint_prob=opt.drop_joint_prob)
    test_dataset = Human36M(actions=actions, data_path=opt.data_dir,
                             dataset_normalization=opt.dataset_normalization, is_train=False)

    if opt.use_horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                        rank=hvd.rank())
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.train_batch,
            sampler=train_sampler, # shuffle=True,#sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(),
                                                                      rank=hvd.rank())
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opt.train_batch,
            sampler=test_sampler, # shuffle=True,#sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.workers,
            pin_memory=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.workers,
            pin_memory=True)
    return train_dataset, train_loader, test_loader





def _train(train_loader, model, criterion, optimizer, writer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          no_grad_clipping=False, grad_clip=10.0, use_horovod=False, tb_log_freq=100):
    """
    A training epoch for the 3D baseline (training via regression only)
    """
    losses = utils.AverageMeter()

    model.train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    for i, (inps, tars, meta) in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.data[0], inputs.size(0))
        loss.backward()
        if not no_grad_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # tensorboard logs
        if glob_step % tb_log_freq == 0:
            writer.add_scalar('data/iter/loss', loss, glob_step)
            weight_mag = parameter_magnitude(model)
            grad_mag = gradient_magnitude(model)
            update_mag = update_magnitude(model, lr_now, grad_mag)
            update_rat = update_ratio(model, lr_now, weight_mag, grad_mag)
            writer.add_scalar('model/weight_magnitude', weight_mag, glob_step)
            writer.add_scalar('model/gradient_magnitude', grad_mag, glob_step)
            writer.add_scalar('model/update_magnitude', update_mag, glob_step)
            writer.add_scalar('model/update_ratio', update_rat, glob_step)

        # update summary
        if (i + 1) % tb_log_freq == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
        if use_horovod and hvd.rank() == 0:
            print('({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
                    .format(batch=i + 1,
                            size=len(train_loader),
                            batchtime=batch_time * 10.0,
                            ttl=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg))

    bar.finish()
    return glob_step, lr_now, losses.avg


def _train_gan(train_loader, G, F, D_X, D_Y, unorm3d, renorm2d, criterion, gen_optimizer, discr_optimizer,
              discr_updates_per_gen_update, lr_init, lr_now, glob_gen_step, glob_discr_step, lr_decay,
              gamma, no_grad_clipping, grad_clip, writer, tb_log_freq, use_horovod,
               gan_coeff, cycle_coeff, regression_coeff, gradient_penalty_coeff, using_projection):
    """
    A training epoch, when training with Cycle GAN.
    Commented out because currently unused. (But some code could be useful at some point and don't want to delete).
    """
    pass
    # losses = utils.AverageMeter()
    #
    # # make sure all models are in train mode
    # G.train()
    # F.train()
    # D_X.train()
    # D_Y.train()
    #
    # # declar losses (to avoid errors)
    # twod_discr_loss = 0.0
    # threed_discr_loss = 0.0
    # twod_grad_penalty = 0.0
    # threed_grad_penalty = 0.0
    # twod_gen_loss = 0.0
    # threed_gen_loss = 0.0
    # twod_cycle_loss = 0.0
    # threed_cycle_loss = 0.0
    #
    # # stats
    # glob_step = glob_discr_step + glob_gen_step
    # start = time.time()
    # batch_time = 0
    # bar = Bar('>>>', fill='>', max=len(train_loader))
    #
    # for i, (inps, tars) in enumerate(train_loader):
    #     if glob_step % lr_decay == 0 or glob_step == 1:
    #         _      = utils.lr_decay(discr_optimizer, glob_step, lr_init, lr_decay, gamma)
    #         lr_now = utils.lr_decay(gen_optimizer, glob_step, lr_init, lr_decay, gamma)
    #
    #     # Perform a step opf training
    #     inputs = Variable(inps.cuda())
    #     targets = Variable(tars.cuda(async=True))
    #     if (i+1) % (discr_updates_per_gen_update+1) == 0:
    #         glob_discr_step += 1
    #         glob_step += 1
    #         twod_discr_loss, threed_discr_loss, twod_grad_penalty, threed_grad_penalty = \
    #             _train_gan_discr_step(inputs, targets, G, F, D_X, D_Y, unorm3d, renorm2d, discr_optimizer, gan_coeff,
    #                                   gradient_penalty_coeff, using_projection)
    #     else:
    #         glob_gen_step += 1
    #         glob_step += 1
    #         twod_gen_loss, threed_gen_loss, twod_cycle_loss, threed_cycle_loss, twod_regression_loss, threed_regression_loss = \
    #             _train_gan_gen_step(inputs, targets, G, F, D_X, D_Y, unorm3d, renorm2d, criterion, gen_optimizer,
    #                                 gan_coeff, cycle_coeff, regression_coeff, using_projection)
    #
    #     # tensorboard logs
    #     if glob_step % tb_log_freq == 0:
    #         writer.add_scalar('gan/2d/discr_loss', twod_discr_loss, glob_step)
    #         writer.add_scalar('gan/2d/gen_loss', twod_gen_loss, glob_step)
    #         writer.add_scalar('gan/2d/cycle_loss', twod_cycle_loss, glob_step)
    #         writer.add_scalar('gan/2d/regression_loss', twod_regression_loss, glob_step)
    #         writer.add_scalar('gan/2d/gradient_penalty', twod_grad_penalty, glob_step)
    #         writer.add_scalar('gan/3d/discr_loss', threed_discr_loss, glob_step)
    #         writer.add_scalar('gan/3d/gen_loss', threed_gen_loss, glob_step)
    #         writer.add_scalar('gan/3d/cycle_loss', threed_cycle_loss, glob_step)
    #         writer.add_scalar('gan/3d/regression_loss', threed_regression_loss, glob_step)
    #         writer.add_scalar('gan/3d/gradient_penalty', threed_grad_penalty, glob_step)
    #
    #         G_weight_mag = parameter_magnitude(G)
    #         G_grad_mag = gradient_magnitude(G)
    #         G_update_mag = update_magnitude(G, lr_now, G_grad_mag)
    #         G_update_rat = update_ratio(G, lr_now, G_weight_mag, G_grad_mag)
    #         writer.add_scalar('2d_to_3d/weight_mag', G_weight_mag, glob_step)
    #         writer.add_scalar('2d_to_3d/grad_mag', G_grad_mag, glob_step)
    #         writer.add_scalar('2d_to_3d/update_mag', G_update_mag, glob_step)
    #         writer.add_scalar('2d_to_3d/update_ratio', G_update_rat, glob_step)
    #
    #         F_weight_mag = parameter_magnitude(F)
    #         F_grad_mag = gradient_magnitude(F)
    #         F_update_mag = update_magnitude(F, lr_now, F_grad_mag)
    #         F_update_rat = update_ratio(F, lr_now, F_weight_mag, F_grad_mag)
    #         writer.add_scalar('3d_to_2d/weight_mag', F_weight_mag, glob_step)
    #         writer.add_scalar('3d_to_2d/grad_mag', F_grad_mag, glob_step)
    #         writer.add_scalar('3d_to_2d/update_mag', F_update_mag, glob_step)
    #         writer.add_scalar('3d_to_2d/update_ratio', F_update_rat, glob_step)
    #
    #         DY_weight_mag = parameter_magnitude(D_Y)
    #         DY_grad_mag = gradient_magnitude(D_Y)
    #         DY_update_mag = update_magnitude(D_Y, lr_now, DY_grad_mag)
    #         DY_update_rat = update_ratio(D_Y, lr_now, DY_weight_mag, DY_grad_mag)
    #         writer.add_scalar('3d_discr/weight_mag', DY_weight_mag, glob_step)
    #         writer.add_scalar('3d_discr/grad_mag', DY_grad_mag, glob_step)
    #         writer.add_scalar('3d_discr/update_mag', DY_update_mag, glob_step)
    #         writer.add_scalar('3d_discr/update_ratio', DY_update_rat, glob_step)
    #
    #         DX_weight_mag = parameter_magnitude(D_X)
    #         DX_grad_mag = gradient_magnitude(D_X)
    #         DX_update_mag = update_magnitude(D_X, lr_now, DX_grad_mag)
    #         DX_update_rat = update_ratio(D_X, lr_now, DX_weight_mag, DX_grad_mag)
    #         writer.add_scalar('2d_discr/weight_mag', DX_weight_mag, glob_step)
    #         writer.add_scalar('2d_discr/grad_mag', DX_grad_mag, glob_step)
    #         writer.add_scalar('2d_discr/update_mag', DX_update_mag, glob_step)
    #         writer.add_scalar('2d_discr/update_ratio', DX_update_rat, glob_step)
    #
    #
    #     # Calculate + log loss (for the "classical regression" task of 2D -> 3D
    #     losses.update(threed_regression_loss.data[0], inputs.size(0))
    #
    #     # update summary
    #     if (i + 1) % 100 == 0:
    #         batch_time = time.time() - start
    #         start = time.time()
    #
    #     bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
    #         .format(batch=i + 1,
    #                 size=len(train_loader),
    #                 batchtime=batch_time * 10.0,
    #                 ttl=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg)
    #     bar.next()
    #     if use_horovod and hvd.rank() == 0:
    #         print('({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
    #                 .format(batch=i + 1,
    #                         size=len(train_loader),
    #                         batchtime=batch_time * 10.0,
    #                         ttl=bar.elapsed_td,
    #                         eta=bar.eta_td,
    #                         loss=losses.avg))
    #
    # return glob_gen_step, glob_discr_step, lr_now, losses.avg



# def _train_gan_discr_step(x, y, G, F, D_X, D_Y, unorm3d, renorm2d, discr_optimizer, gan_coeff, gradient_penalty_coeff,
#                           using_projection):
#     """
#     A training step for the discriminator, when training with Cycle GAN.
#     Commented out because currently unused. (But some code could be useful at some point and don't want to delete).
#     """
#     # x = 2D pose, y = 3D pose, x_G = G(x), x_G_F = G(F(x)) and so on
#     x_G = G(x)
#     y_F = renorm2d(F(unorm3d(y))) if using_projection else F(y)
#
#     discr_optimizer.zero_grad()
#
#     # normal discriminator losses in 2D and 3D
#     twod_discr_loss = torch.mean((D_X(y_F) - 1.0) ** 2)
#     threed_discr_loss = torch.mean((D_Y(x_G) - 1.0) ** 2)
#
#     # Gradient penalty, compute interpolation between real and false data points
#     batch_size = x.size(0)
#
#     alpha_3d = torch.rand(batch_size, 1)
#     alpha_3d = alpha_3d.expand_as(y).cuda()
#     inter_3d = alpha_3d * y.data + (1 - alpha_3d) * x_G.data
#     inter_3d = Variable(inter_3d, requires_grad=True).cuda()
#
#     alpha_2d = torch.rand(batch_size, 1)
#     alpha_2d = alpha_2d.expand_as(x).cuda()
#     inter_2d = alpha_2d * x.data + (1 - alpha_2d) * y_F.data
#     inter_2d = Variable(inter_2d, requires_grad=True).cuda()
#
#     # Compute gradient of discriminator at "interpolated", and use it for gradient penalty, as 1e-12 for numerical stability
#     inter_3d_prob = D_Y(inter_3d)
#     gradients_discr_3d = torch.autograd.grad(outputs=inter_3d_prob, inputs=inter_3d,
#                            grad_outputs=torch.ones(inter_3d_prob.size()).cuda(),
#                            create_graph=True, retain_graph=True)[0]
#     gradients_norm_discr_3d = torch.sqrt(torch.sum(gradients_discr_3d ** 2, dim=1) + 1e-12)
#     threed_grad_penalty = torch.mean(torch.sqrt((gradients_norm_discr_3d - 1.0) ** 2))
#
#     inter_2d_prob = D_X(inter_2d)
#     gradients_discr_2d = torch.autograd.grad(outputs=inter_2d_prob, inputs=inter_2d,
#                            grad_outputs=torch.ones(inter_2d_prob.size()).cuda(),
#                            create_graph=True, retain_graph=True)[0]
#     gradients_norm_discr_2d = torch.sqrt(torch.sum(gradients_discr_2d ** 2, dim=1) + 1e-12)
#     twod_grad_penalty = torch.mean(torch.sqrt((gradients_norm_discr_2d - 1.0) ** 2))
#
#     # accuulate all the losses nad make a step
#     discr_loss = gan_coeff * (twod_discr_loss + threed_discr_loss) \
#                     + gradient_penalty_coeff * (twod_grad_penalty + threed_grad_penalty)
#     discr_loss.backward()
#     discr_optimizer.step()
#
#     # Return the constituent losses for plotting
#     return twod_discr_loss, threed_discr_loss, twod_grad_penalty, threed_grad_penalty



# def _train_gan_gen_step(x, y, G, F, D_X, D_Y, unorm3d, renorm2d, criterion, gen_optimizer, gan_coeff, cycle_coeff,
#                         regression_coeff, using_projection):
#     """
#     A training step for the generator, when training with Cycle GAN.
#     Commented out because currently unused. (But some code could be useful at some point and don't want to delete).
#     """
#     # x = 2D pose ground truth, y = 3D pose ground truch, x_G = G(x), x_G_F = G(F(x)) and so on
#     x_G = G(x)
#     x_G_F = renorm2d(F(unorm3d(x_G))) if using_projection else F(x_G)
#     y_F = renorm2d(F(unorm3d(y))) if using_projection else F(y)
#     y_F_G = G(y_F)
#
#     gen_optimizer.zero_grad()
#
#     twod_gen_loss = torch.mean(D_X(y_F) ** 2)
#     threed_gen_loss = torch.mean(D_Y(x_G) ** 2)
#     twod_cycle_loss = torch.sum(torch.mean(torch.abs(x - x_G_F), dim=0))
#     threed_cycle_loss = torch.sum(torch.mean(torch.abs(y - y_F_G), dim=0))
#     threed_regression_loss = criterion(x_G, y)
#     twod_regression_loss = criterion(y_F, x)
#
#     gen_loss = gan_coeff * (twod_gen_loss + threed_gen_loss) \
#                + cycle_coeff * (twod_cycle_loss + threed_cycle_loss) \
#                + regression_coeff * (twod_regression_loss + threed_regression_loss)
#
#     gen_loss.backward()
#     gen_optimizer.step()
#
#     return twod_gen_loss, threed_gen_loss, twod_cycle_loss, threed_cycle_loss, twod_regression_loss, threed_regression_loss



def _test(test_loader, model, criterion, dataset_normalization, procrustes=False):
    """
    A validation epoch, to test the prediction of 3D poses from 2D poses, regardless of how the network has been trained.
    Because really, this is all that we care about.
    """
    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, (inps, tars, meta) in enumerate(test_loader):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets)

        losses.update(loss.data[0], inputs.size(0))

        # Calculate the errors in the unormalized space
        all_dist.append(compute_3d_pose_error_distances(outputs, targets, meta, dataset_normalization, procrustes))

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err





if __name__ == "__main__":
    option = Options().parse()
    main(option)
