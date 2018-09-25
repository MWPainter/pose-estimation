from torch.utils.data import Dataset, DataLoader

from generative_models import FullyConnectedGan

from twod_threed.src.datasets.human36m import Human36M
import twod_threed.src.misc as misc

from utils import train_loop
from utils import parameter_magnitude, gradient_magnitude, update_magnitude, update_ratio





class Human36M3DPoseDataset(Dataset):
    """
    Wrapper around the Human36M dataset, that just provides 3d poses
    """
    def __init__(self, actions, data_path, is_train=True):
        self.h36m_dataset = Human36M(actions, data_path, False, is_train)

    def __getitem__(self, index):
        return self.h36m_dataset[index][1]

    def __len__(self):
        return len(self.h36m_dataset)







def _make_optimizer_fn(model, lr, weight_decay):
    """
    The make optimizer function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to make an optimizer for (in this case a FCGan object)
    :param lr: The learning rate to use
    :param weight_decay: THe weight decay to use
    :return: The optimizer for the network (discr_optimizer, gen_optimizer), which is passed into the remaining
        training loop functions
    """
    discriminator_optimizer = torch.optim.Adam(model.discr.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    generator_optimizer = torch.optim.Adam(model.gen.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    return discriminator_optimizer, generator_optimizer


def _load_fn(model, optimizer, load_file):
    """
    The make load (model) function, as part of the interface for the "train_loop" function in utils.training_utils.

    :param model: The model to restore the state for
    :param optimizer: The optimizer to restore the state for (same as the return value from 'make_optimizer_fn')
    :param load_file: The filename for a checkpoint dict, saved using 'checkpoint_fn' below.
    :return: The restored model, optimizer and the current epoch with the best validation loss seen so far.
    """
    # Unpack optimizer
    discriminator_optimizer, generator_optimizer = optimizer

    # Load state dict, and update the model and
    checkpoint = torch.load(load_file)
    cur_epoch = checkpoint['next_epoch']
    best_val_loss = checkpoint['best_val_loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['discr_optimizer_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])

    # Return the model and optimizer with restored state
    return model, (discriminator_optimizer, generator_optimizer), cur_epoch, best_val_loss


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
    # Unpack
    discriminator_optimizer, generator_optimizer = optimizer

    # Make the checkpoint
    checkpoint = {}
    checkpoint['next_epoch'] = epoch + 1
    checkpoint['best_val_loss'] = best_val_loss
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['discr_optimizer_state_dict'] = discriminator_optimizer.state_dict()
    checkpoint['gen_optimizer_state_dict'] = generator_optimizer.state_dict()

    # Save it as the most up to date checkpoint
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(checkpoint, filename)

    # Save it as the "best" checkpoint if we are the best
    if is_best_so_far:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(checkpoint, best_filename)


def _update_op(model, optimizer, minibatch, iter, args):
    """
    The update op function, as part of the interface for the "train_loop" function in utils.training_utils.
    This function directly *performs* the update on model parameters. A number of losses will be computed,
    and are returned in a dictionary of PyTorch scalar Variables. The dictionary is used to log TensorBoard
    summaries, by the main training loop.

    The WGAN update will alternate between updating the discriminator and generator. Parameter clipping and
    gradient penalties are applied to the discriminator, ontop of a non-saturating Gan loss.

    :param model: The Gan nn.Module object
    :param optimizer: The optimizer object (as defined by make_optimizer_fn)
    :param minibatch: A minibatch of data to use for this update
    :param iter: The iteration index
    :param args: The command line arguments (opt parser) passed in through the command line.
    :return: A dictionary from strings to PyTorch scalar Variables used for TensorBoard summaries.
    """
    # Unpack
    n_discr = args.num_discriminator_steps_per_generator_step + 1
    discriminator_optimizer, generator_optimizer = optimizer

    # If minibatch is of shape (N,D), then we should sample N random samples from the generator
    data = minibatch.cuda()
    N, D = list(data.size())
    generator_samples = model.sample(N)

    # Compute the generator or discriminator loss, and make an appropriate step, clipping discriminator weights
    losses = {}
    if iter % n_discr != 0:
        grad_penalty = _grad_penalty(N, data, generator_samples, model.discr)
        gan_discr_loss = - torch.mean(torch.log(model.discr(data))) \
                            - torch.mean(1.0 - torch.log(model.discr(generator_samples)))
        total_discr_loss = gan_discr_loss
        if args.gradient_penalty_coeff > 0.0: total_discr_loss += args.gradient_penalty_coeff * grad_penalty
        losses['discr/grad_penalty'] = grad_penalty
        losses['disrc/gan_discr_loss'] = gan_discr_loss
        losses['discr/total_discr_loss'] = total_discr_loss

        discriminator_optimizer.zero_grad()
        total_discr_loss.backward()
        discriminator_optimizer.step()

        for p in model.discr.parameters():
            p.data.clamp_(-model.clip_max, model.clip_max)

    else:
        gen_loss = -torch.mean(torch.log(model.discr(generator_samples)))
        losses['gen/gen_loss'] = gen_loss
        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()

    # Losses about weight norms etc. Only compute occasionally because this is heavyweight
    if iter % args.tb_log_freq == 0:
        gen_weight_mag = parameter_magnitude(model.gen)
        gen_grad_mag = gradient_magnitude(model.gen)
        gen_update_mag = update_magnitude(model.gen, args.lr, gen_grad_mag)
        gen_update_ratio = update_ratio(model.gen, args.lr, gen_weight_mag, gen_grad_mag)
        discr_weight_mag = parameter_magnitude(model.discr)
        discr_grad_mag = gradient_magnitude(model.discr)
        discr_update_mag = update_magnitude(model.discr, args.lr, discr_grad_mag)
        discr_update_ratio = update_ratio(model.discr, args.lr, discr_weight_mag, discr_grad_mag)
        losses['gen/weight_mag'] = gen_weight_mag
        losses['gen/grad_mag'] = gen_grad_mag
        losses['gen/update_mag'] = gen_update_mag
        losses['gen/update_ratio'] = gen_update_ratio
        losses['discr/weight_mag'] = discr_weight_mag
        losses['discr/grad_mag'] = discr_grad_mag
        losses['discr/update_mag'] = discr_update_mag
        losses['discr/update_ratio'] = discr_update_ratio

    # Return the dictionary of 'losses'
    return losses


def _grad_penalty(N, data, samples, discr):
    """
    Helper function to compute the gradient penalty from [https://arxiv.org/pdf/1704.00028].

    :param N: The number of samples to use for the gradient penalty.
    :param data: A PyTorch Variable, of real data from the dataset. With shape (N, D) if D is the data size.
    :param samples: A PyTorch Variable, of data generated from the Generator. With the same shape (N, D) as 'data'.
    :param discr: The discriminator network for the GAN
    :return: The gradient penalty (A scalar PyTorch Variable)
    """
    # Interpolate between the generator samples and true data
    alpha = torch.rand(N, 1)
    alpha = alpha.expand_as(data).cuda()
    inter = alpha * data.data + (1-alpha) * samples.data
    inter = torch.autograd.Variable(inter, requires_grad=True).cuda()

    # Compute gradient of discriminator at "interpolated", and use it for gradient penalty, 1e-12 for numerical stability
    probs = discr(inter)
    gradients = torch.autograd.grad(outputs=probs, inputs=inter,
                                             grad_outputs=torch.ones(probs.size()).cuda(),
                                             create_graph=True, retain_graph=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return torch.mean(torch.sqrt((gradients_norm - 1.0) ** 2))


def validation_loss(model, minibatch):
    """
    Computes the (non-saturating) GAN loss on a minibatch that has not been seen during training at all.

    :param model: The GAN model to compute the validation loss for.
    :param minibatch: A PyTorch Varialbe of shape (N,D) to compute the validation loss over.
    :return: A PyTorch scalar Variable with value of the validation loss.
        Returns the discriminator and generator losses.
    """
    # Put in eval mode
    model.eval()

    # Compute samples
    data = minibatch.cuda()
    N, D = minibatch.size()
    gen_samples = model.sample(N)

    # Compute the generator and discriminator losses
    discr_loss = - torch.mean(torch.log(model.discr(data))) \
                            - torch.mean(torch.log(1.0 - model.discr(gen_samples)))
    gen_loss = -torch.mean(torch.log(model.discr(gen_samples)))

    # Return the dictionary of losses
    return {'discriminator_loss': discr_loss,
            'generator_loss': gen_loss}






def train_3d_pose_gan(args):
    """
    Train a GAN for 3D poses, with arguments 'args'. This just makes some data loaders, using the 3D Pose dataset object
    above. Really this just calls the train_loop function from utilz.train_utils.

    :param args: Arguments from an ArgParser specifying how to run the trianing
    """
    actions = misc.define_actions(args.action)

    train_dataset = Human36M3DPoseDataset(actions=actions, data_path=args.data_dir, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_dataset = Human36M3DPoseDataset(actions=actions, data_path=args.data_dir, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.test_batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    model = FullyConnectedGan(clip_max=args.clip_max).cuda()

    train_loop(model, train_loader, val_loader, _make_optimizer_fn, _load_fn, _checkpoint_fn, _update_op,
               _validation_loss, args)



