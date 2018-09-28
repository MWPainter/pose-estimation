from torch.utils.data import Dataset, DataLoader

from stitched.stitched_network import StitchedNetwork
from stitched.run import load_model_and_dataset_mpii

from utils.human36m_dataset import Human36mDataset

from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio





def _make_optimizer_fn(model, lr, weight_decay):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to make an optimizer for (in this case a FCGan object)
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :return: The optimizer for the network, which is passed into the remaining
        training loop functions
    """
    return torch.optim.Adam(model.discr.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)


def _load_fn(model, optimizer, load_file):
    """
    The make load (model) function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to restore the state for
    :param optimizer: The optimizer to restore the state for (same as the return value from 'make_optimizer_fn')
    :param load_file: The filename for a checkpoint dict, saved using 'checkpoint_fn' below.
    :return: The restored model, optimizer and the current epoch with the best validation loss seen so far.
    """
    # Load state dict, and update the model and
    checkpoint = torch.load(load_file)
    cur_epoch = checkpoint['next_epoch']
    best_val_loss = checkpoint['best_val_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the model and optimizer with restored state
    return model, optimizer, cur_epoch, best_val_loss


def _checkpoint_fn(model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best_so_far):
    """
    The checkpoint function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function will take the current state of training (i.e. the tuple (model, optimizer, epoch, best_val_loss)
    save it in the appropriate checkpoint file(s), in

    :param model: The GAN model to take the checkpoint for
    :param optimizer: The optimizer to take the checkpoint for
    :param epoch: The current epoch in training
    :param best_val_loss: The best validation loss seen so far
    :param checkpoint_dir: The directory for which to save checkpoints in
    :param is_best_so_far: If the checkpoint is the best so far (with respect to the validation loss)
    :return: Nothing.
    """
    # Make the checkpoint
    checkpoint = {}
    checkpoint['next_epoch'] = epoch + 1
    checkpoint['best_val_loss'] = best_val_loss
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save it as the most up to date checkpoint
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(checkpoint, filename)

    # Save it as the "best" checkpoint if we are the best
    if is_best_so_far:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(checkpoint, best_filename)


def _update_op(model, optimizer, minibatch_data, iter, args):
    """
    The update op function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function directly *performs* the update on model parameters. A number of losses will be computed,
    and are returned in a dictionary of PyTorch scalar Variables. The dictionary is used to log TensorBoard
    summaries, by the main training loop.

    The WGAN update will alternate between updating the discriminator and generator. Parameter clipping and
    gradient penalties are applied to the discriminator, ontop of a non-saturating Gan loss.

    :param model: The Gan nn.Module object
    :param optimizer: The optimizer object (as defined by make_optimizer_fn)
    :param minibatch_data: One sampling from the PyTorch data loader. In this case it will be a tuple
        (img_for_hg_input, target_heatmap, normalized_pose_2d, normalized_pose_3d, meta)
    :param iter: The iteration index
    :param args: The command line arguments (opt parser) passed in through the command line.
    :return: A dictionary from strings to PyTorch scalar Variables used for TensorBoard summaries.
    """
    # Change to training mode
    model.train()

    # Unpack the minibatch data, and, run the forward pass (get all of the data to compute a loss)
    img, _, _, pose, meta = minibatch_data

    inputs = Variable(inps.cuda())
    targets = Variable(pose.cuda(async=True))
    criterion = nn.MSELoss(size_average=True).cuda()

    preds = model(inputs)

    # Compute the loss, and make an optimization step
    losses = {}
    loss = criterion(targets, preds)
    losses["iter/loss"] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Losses about weight norms etc. Only compute occasionally because this is heavyweight
    if iter % args.tb_log_freq == 0:
        weight_mag = parameter_magnitude(model)
        grad_mag = gradient_magnitude(model)
        update_mag = update_magnitude(model, args.lr, grad_mag)
        update_ratio = update_ratio(model, args.lr, weight_mag, grad_mag)
        losses['model/weight_mag'] = weight_mag
        losses['model/grad_mag'] = grad_mag
        losses['model/update_mag'] = update_mag
        losses['model/update_ratio'] = update_ratio

    # Return the dictionary of 'losses'
    return losses


def validation_loss(model, minibatch_data):
    """
    Computes the (non-saturating) GAN loss on a minibatch that has not been seen during training at all.

    :param model: The GAN model to compute the validation loss for.
    :param minibatch: One sampling from the PyTorch data loader. In this case it will be a tuple
        (img_for_hg_input, target_heatmap, normalized_pose_2d, normalized_pose_3d, meta)
    :return: A PyTorch scalar Variable with value of the validation loss.
        Returns the discriminator and generator losses.
    """
    # Put in eval mode
    model.eval()

    # Unpack the minibatch data, and, run the forward pass (get all of the data to compute a loss)
    img, _, _, pose, meta = minibatch_data

    inputs = Variable(inps.cuda())
    targets = Variable(pose.cuda(async=True))
    criterion = nn.MSELoss(size_average=True).cuda()

    preds = model(inputs)

    # Compute the same loss as above
    loss = criterion(targets, preds)

    # Unnormalize, and compute a full error
    errs = data_utils.compute_3d_pose_error_distances(preds, targets, meta, model.baseline_dataset_normalization)
    # TODO: add a per joint error here
    err = np.mean(errs)

    # Return the dictionary of losses
    return {'loss': loss,
            'err': err}






def train_stitched_fine_tune(args):
    """
    Fine tune the stitched network, with arguments 'args'. This just makes some data loaders, using the 3D Pose dataset object
    above. Really this just calls the train_loop function from utilz.train_utils.

    We assume that the stacked hourglass network was pre-trained on MPII dataset, so we use the color normalization
    from this dataset.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    # Unpack options
    hg_model_file = args.load_hourglass
    threed_baseline_model_file = args.load_2d3d
    data_input_dir = args.data_dir
    data_output_dir = args.output_dir
    dataset_normalization = args.dataset_normalization

    # Create the model, and load the pre-trained subnetworks (loading is more complex, because the stitched
    # network needs to be initialized with the correct color means etc). So just re-use from the run.py
    # TODO: refactor this, so it's somewhere else, and used in both train and run. HAve a stitched/utils.py probs
    model, _ = load_model_and_dataset_mpii(hg_model_file, threed_baseline_model_file, data_input_dir, args)
    model.train()

    # Make data loaders, correcting the color norm and std (as the hourglass was pre-trained on MPII)
    train_dataset = Human36mDataset(dataset_path=data_input_dir, is_train=True,
                                    dataset_normalization=dataset_normalization, load_image_data=True)
    train_dataset.set_color_mean(model.hg_mean)
    train_dataset.set_color_std(model.hg_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_dataset = Human36mDataset(dataset_path=data_input_dir, is_train=False,
                                  dataset_normalization=dataset_normalization, load_image_data=True)
    val_dataset.set_color_mean(model.hg_mean)
    val_dataset.set_color_std(model.hg_std)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.test_batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Run the training loop
    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)



