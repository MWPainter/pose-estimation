from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import torch.utils.data.distributed
import horovod.torch as hvd

from stacked_hourglass.pose import Bar
from stacked_hourglass.pose.utils.logger import Logger, savefig
from stacked_hourglass.pose.utils.evaluation import accuracy_PCK, accuracy_PCKh, final_preds
from stacked_hourglass.pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from stacked_hourglass.pose.utils.imutils import batch_with_heatmap
from stacked_hourglass.pose.utils.transforms import fliplr, flip_back
from stacked_hourglass.pose.models import HourglassNet, JointVisibilityNet
import stacked_hourglass.pose.datasets as datasets

from tensorboardX import SummaryWriter

from utils import AverageMeter
from utils import count_parameters, parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0


def main(args):
    """
    Main training loop for training a stacked hourglass model on MPII dataset.
    :param args: Command line arguments.
    """
    global best_acc

    # create checkpoint dir
    if not isdir(args.checkpoint_dir):
        mkdir_p(args.checkpoint_dir)

    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = HourglassNet(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes,
                         batch_norm_momentum=args.batch_norm_momentum, use_layer_norm=args.use_layer_norm, width=256, height=256)
    joint_visibility_model = JointVisibilityNet(hourglass_stacks=args.stacks)

    # scale weights
    if args.scale_weight_factor != 1.0:
        model.scale_weights_(args.scale_weight_factor)

    # setup horovod and model for parallel execution
    if args.use_horovod:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        args.lr *= hvd.size()
        model.cuda()
    else:
        model = model.cuda()
        if args.predict_joint_visibility:
            joint_visibility_model = joint_visibility_model.cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    joint_visibility_criterion = None if not args.predict_joint_visibility else torch.nn.BCEWithLogitsLoss()
    params = [{'params': model.parameters(), 'lr': args.lr}]
    if args.predict_joint_visibility:
        params.append({'params': joint_visibility_model.parameters(), 'lr': args.lr})
    params = model.parameters()
    if not args.use_amsprop:
        optimizer = torch.optim.RMSprop(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params,
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     amsgrad=True)
    if args.use_horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Create a tensorboard writer
    writer = SummaryWriter(log_dir="%s/hourglass_mpii_%s_tb_log" % (args.tb_dir, args.exp))

    # optionally resume from a checkpoint
    title = 'mpii-' + args.arch
    if args.load:
        if isfile(args.load):
            print("=> loading checkpoint '{}'".format(args.load))
            checkpoint = torch.load(args.load)

            # remove old usage of data parallel (used to be wrapped around model) # TODO: remove this when no old models used this
            state_dict = {}
            for key in checkpoint['state_dict']:
                new_key = key[len("module."):] if key.startswith("module.") else key
                state_dict[new_key] = checkpoint['state_dict'][key]

            # restore state
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(state_dict)
            if args.predict_joint_visibility: joint_visibility_model.load_state_dict(checkpoint['joint_visibility_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint_dir, 'log.txt'), title=title, resume=True)
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.load))
    else:        
        logger = Logger(join(args.checkpoint_dir, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    train_dataset, train_loader, val_loader = _make_torch_data_loaders(args)

    if args.evaluate:
        print('\nEvaluation only') 
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint_dir)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc, joint_visibility_loss, joint_visibility_acc = train(train_loader, model=model,
                                      joint_visibility_model=joint_visibility_model, criterion=criterion, num_joints=args.num_classes,
                                      joint_visibility_criterion=joint_visibility_criterion, optimizer=optimizer,
                                      epoch=epoch, writer=writer, lr=lr, debug=args.debug, flip=args.flip,
                                      remove_intermediate_supervision=args.remove_intermediate_supervision,
                                      tb_freq=args.tb_log_freq, no_grad_clipping=args.no_grad_clipping,
                                      grad_clip=args.grad_clip, use_horovod=args.use_horovod,
                                      predict_joint_visibility=args.predict_joint_visibility,
                                      predict_joint_loss_coeff=args.joint_visibility_loss_coeff)

        # evaluate on validation set
        valid_loss, valid_acc_PCK, valid_acc_PCKh, valid_acc_PCKh_per_joint, valid_joint_visibility_loss, valid_joint_visibility_acc, predictions = validate(
                                        val_loader, model, joint_visibility_model, criterion, joint_visibility_criterion, args.num_classes, args.debug, args.flip,
                                        args.use_horovod, args.use_train_mode_to_eval, args.predict_joint_visibility)

        # append logger file, and write to tensorboard summaries
        writer.add_scalars('data/epoch/losses_wrt_epochs', {'train_loss': train_loss, 'test_lost': valid_loss}, epoch)
        writer.add_scalar('data/epoch/train_accuracy_PCK', train_acc, epoch)
        writer.add_scalar('data/epoch/test_accuracy_PCK', valid_acc_PCK, epoch)
        writer.add_scalar('data/epoch/test_accuracy_PCKh', valid_acc_PCKh, epoch)
        for key in valid_acc_PCKh_per_joint:
            writer.add_scalar('per_joint_data/epoch/test_accuracy_PCKh_%s' % key, valid_acc_PCKh_per_joint[key], epoch)
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc_PCK])
        if args.predict_joint_visibility:
            writer.add_scalars('joint_visibility/epoch/loss', {'train': joint_visibility_loss, 'test_lost': valid_joint_visibility_loss}, epoch)
            writer.add_scalars('joint_visibility/epoch/acc', {'train': joint_visibility_acc, 'test_lost': valid_joint_visibility_acc}, epoch)

        # remember best acc and save checkpoint
        model_specific_checkpoint_dir = "%s/hourglass_mpii_%s" % (args.checkpoint_dir, args.exp)
        if not isdir(model_specific_checkpoint_dir):
            mkdir_p(model_specific_checkpoint_dir)

        is_best = valid_acc_PCK > best_acc
        best_acc = max(valid_acc_PCK, best_acc)
        mean, stddev = train_dataset.get_mean_stddev()
        checkpoint = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'mean': mean,
            'stddev': stddev,
        }
        if args.predict_joint_visibility:
            checkpoint['joint_visibility_state_dict'] = joint_visibility_model.state_dict()
        save_checkpoint(checkpoint, predictions, is_best, checkpoint=model_specific_checkpoint_dir)

    logger.close()
    #logger.plot(['Train Acc', 'Val Acc'])
    #savefig(os.path.join(args.checkpoint_dir, 'log.eps'))





def _make_torch_data_loaders(args):
    """
    Helper function to produce data loaders for training
    :param args: Command line arguments
    :return: Training dataset, training data loader, validation data loader.
    """
    train_dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json',
                                  'stacked_hourglass/data/mpii/images',
                                  sigma=args.sigma, label_type=args.label_type,
                                  augment_data=args.augment_training_data, args=args)

    val_dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json',
                                'stacked_hourglass/data/mpii/images',
                                sigma=args.sigma, label_type=args.label_type, train=False, augment_data=False, args=args)

    if args.use_horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                        rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size, sampler=train_sampler,
                                                   # shuffle=True,#sampler=train_sampler,
                                                   num_workers=args.workers, pin_memory=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(),
                                                                      rank=hvd.rank())
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.test_batch_size, sampler=val_sampler,
                                                 # shuffle=False, #sampler=val_sampler,
                                                 num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

    return train_dataset, train_loader, val_loader





def train(train_loader, model, joint_visibility_model, criterion, num_joints, joint_visibility_criterion, optimizer,
          epoch, writer, lr, debug=False, flip=True, remove_intermediate_supervision=False, tb_freq=100,
          no_grad_clipping=False, grad_clip=10.0, use_horovod=False, predict_joint_visibility=False,
          predict_joint_loss_coeff=0.0):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    visibility_losses = AverageMeter()
    visibility_accs = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()

    epoch_len = len(train_loader)
    epoch_beg_iter = epoch * epoch_len

    gt_win, pred_win = None, None
    bar = Bar('Processing', max=epoch_len)
    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        # compute output
        output = model(input_var)
        score_map = output[-1].data.cpu()

        # Add losses (only add final loss if ignoring intermediate supervision) + compute end accuracy
        loss = criterion(output[len(output)-1], target_var)
        if not remove_intermediate_supervision:
            for j in range(len(output)-2, -1, -1):
                loss += criterion(output[j], target_var)
        acc = accuracy_PCK(score_map, target, idx)

        # Add joint visibility loss if necessary
        if predict_joint_visibility:
            visibility_input = torch.stack(output, dim=2).view(inputs.size(0) * num_joints, -1)
            visibility_gts = _joint_visibility_ground_truths_from_meta(meta)
            visibility_pred_logits = joint_visibility_model(visibility_input)
            visibility_loss = joint_visibility_criterion(visibility_pred_logits, visibility_gts)
            visibility_acc = _joint_visibility_acc(visibility_pred_logits, visibility_gts)
            loss += predict_joint_loss_coeff * visibility_loss

        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        if predict_joint_visibility:
            visibility_losses.update(visibility_loss.data, inputs.size(0))
            visibility_accs.update(torch.mean(visibility_acc), inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if not no_grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if predict_joint_visibility:
                torch.nn.utils.clip_grad_norm_(joint_visibility_model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # Plot the (noisy) per minibatch loss once every so often
        iter = epoch_beg_iter + i
        if iter % tb_freq == 1:
            weight_mag = parameter_magnitude(model)
            grad_mag = gradient_magnitude(model)
            update_mag = update_magnitude(model, lr, grad_mag)
            update_rat = update_ratio(model, lr, weight_mag, grad_mag)
            writer.add_scalar('iter/hg/train_loss', loss, iter)
            writer.add_scalar('iter/hg/weight_magnitude', weight_mag, iter)
            writer.add_scalar('iter/hg/gradient_magnitude', grad_mag, iter)
            writer.add_scalar('iter/hg/update_magnitude', update_mag, iter)
            writer.add_scalar('iter/hg/update_ratio', update_rat, iter)
            if predict_joint_visibility:
                weight_mag = parameter_magnitude(joint_visibility_model)
                grad_mag = gradient_magnitude(joint_visibility_model)
                update_mag = update_magnitude(joint_visibility_model, lr, grad_mag)
                update_rat = update_ratio(joint_visibility_model, lr, weight_mag, grad_mag)
                writer.add_scalar('joint_visibility/iter/loss', visibility_loss, iter)
                writer.add_scalar('joint_visibility/iter/hg/weight_magnitude', weight_mag, iter)
                writer.add_scalar('joint_visibility/iter/hg/gradient_magnitude', grad_mag, iter)
                writer.add_scalar('joint_visibility/iter/hg/update_magnitude', update_mag, iter)
                writer.add_scalar('joint_visibility/iter/hg/update_ratio', update_rat, iter)


        # plot progress
        prog_str = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.suffix = prog_str
        bar.next()

        # Progress bar seems to not work with horovod?
        if use_horovod and hvd.rank() == 0:
            print(prog_str)

    bar.finish()
    return losses.avg, acces.avg, visibility_losses.avg, visibility_accs.avg





def validate(val_loader, model, joint_visibility_model, criterion, joint_visibility_criterion, num_classes, debug=False, flip=True, use_horovod=False, use_train_mode_to_eval=False, predict_joint_visibility=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces_PCK = AverageMeter()
    acces_PCKh = AverageMeter()
    acces_PCKh_per_joint = defaultdict(AverageMeter)
    visibility_losses = AverageMeter()
    visibility_accs = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    if not use_train_mode_to_eval:
        model.eval()
        if predict_joint_visibility:
            joint_visibility_model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        score_map = output[-1].data.cpu()
        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(),
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # Compute visibilities (reshape the output to (batchsize * numjoints, numstacks * width * height)
        if predict_joint_visibility:
            visibility_input = torch.stack(output, dim=2).view(inputs.size(0)*num_classes, -1)
            visibility_gts = _joint_visibility_ground_truths_from_meta(meta)
            visibility_pred_logits = joint_visibility_model(visibility_input)
            visibility_loss = joint_visibility_criterion(visibility_pred_logits, visibility_gts)
            visibility_acc = _joint_visibility_acc(visibility_pred_logits, visibility_gts)


        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc_PCK = accuracy_PCK(score_map, target.cpu(), idx)
        acc_PCKh, acc_PCKh_per_joint = accuracy_PCKh(score_map, target.cpu(), meta, idx, val_loader.dataset.joint_idxs)

        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]


        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))
        acces_PCK.update(acc_PCK[0], inputs.size(0))
        acces_PCKh.update(acc_PCKh, inputs.size(0))
        for key in acc_PCKh_per_joint:
            acces_PCKh_per_joint[key].update(acc_PCKh_per_joint[key][0], inputs.size(0)) # acc_PCKh_per_joint[key][1])
        if predict_joint_visibility:
            visibility_losses.update(visibility_loss.data, inputs.size(0))
            visibility_accs.update(visibility_acc, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        prog_str = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces_PCK.avg
        )
        bar.suffix = prog_str
        bar.next()

        # Progress bar seems to not work with horovod?
        if use_horovod and hvd.rank() == 0:
            print(prog_str)

    bar.finish()

    PCKh_per_joint = {}
    for key in acces_PCKh_per_joint:
        PCKh_per_joint[key] = acces_PCKh_per_joint[key].avg

    return losses.avg, acces_PCK.avg, acces_PCKh.avg, PCKh_per_joint, visibility_losses.avg, visibility_accs.avg, predictions




def _joint_visibility_acc(logits, ground_truth_visibilities):
    """
    logits = joint visibility prediction logits
    ground_truth_visibilities = ground truths
    Returns an array, where if the ith value is 1, then the ith prediction was correct
    The predictions will be 1.0 for any logit >=0 and 0.0 for any logit <0
    """
    preds = logits.cpu() >= 0
    correct = preds.int() == ground_truth_visibilities.cpu().int()
    return torch.mean(correct.float())




def _joint_visibility_ground_truths_from_meta(meta):
    """
    Get the joint visibilities from the meta variable
    This could be of the shape [batch_size, num_joints, 1], but we will return it as [batch_size * num_joints, 1] with reshaping
    """
    points = meta['pts']
    vis = points[:,:,2]
    batch_size, num_joints = list(vis.size())
    return vis.view(batch_size * num_joints, 1).cuda()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=16, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())