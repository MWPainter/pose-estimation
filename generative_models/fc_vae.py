############################################
# NOT USED (but couldn't face deleting it) #
############################################




# import torch
# import torch.nn as nn
#
# from base_network import TinyMultiTaskResNet
#
#
#
#
#
# class _FCEncoder(nn.Module):
#     def __init__(self, input_size=48, hidden_size=256, latent_size=10, layers_per_block=2, num_blocks=1):
#         super(_FCEncoder, self).__init__()
#         self.base_network = TinyMultiTaskResNet(input_size, hidden_size, [latent_size, latent_size], layers_per_block, num_blocks)
#
#     def forward(self, x):
#         mean_pre_act, std_pre_act = self.base_network(x)
#         mean = 2.0 * F.sigmoid(mean_pre_act) - 1.0
#         std = F.sigmoid(std_pre_act)
#         return mean, std
#
#
#
#
#
# class _FCDecoder(nn.Module):
#     def __init__(self, latent_size=10, hidden_size=256, output_size=48, layers_per_block=2, num_blocks=1):
#         super(_FCDecoder, self).__init__()
#         self.base_network = TinyMultiTaskResNet(latent_size, hidden_size, [output_size], layers_per_block, num_blocks)
#
#     def forward(self, x):
#         return self.base_network(x)[0]
#
#
#
#
#
# class FullyConnectedVAE(nn.Module):
#     def __init__(self, data_size=48, latent_size=10):
#         super(FullyConnectedVAE, self).__init__()
#         self.model_name = "fully_connected_vae"
#
#         self.data_size = data_size
#         self.latent_size = latent_size
#
#         self.encoder = _FCEncoder(input_size=data_size, latent_size=latent_size)
#         self.decoder = _FCDecoder(latent_size=latent_size, output_size=data_size)
#
#         self.normal = torch.distributions.Normal(loc=0, scale=1)
#
#
#     def sample(self, n):
#         """
#         Use the network in generative mode, sampling latent variables and using the decoder.
#
#         :param n: The number of samples desired
#         :return: The samples, returns as a batch with shape (n, self.data_size)
#         """
#         shape = torch.Size((n, self.latent_size))
#         latents = self.normal.sample(shape)
#         return self.decoder(latents)
#
#
#     def forward(self, x):
#         # encode, sample, decode
#         mean, std = self.encoder(x)
#         shape = mean.size()
#         latents = self.normal.sample(shape) * std + mean
#         return self.decoder(latents)
#
#     ###
#     ### Training loop functions
#     ###
#     def make_optimizer_fn(self, model, lr, weight_decay):
#         return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
#
#
#     def load_fn(self, model, optimizer, load_file):
#
#         # Load state dict, and update the model and
#         checkpoint = torch.load(load_file)
#         cur_epoch = checkpoint['next_epoch']
#         best_val_loss = checkpoint['best_val_loss']
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#         # Return the model and optimizer with restored state
#         return model, optimizer, cur_epoch, best_val_loss
#
#
#     def checkpoint_fn(self, model, optimizer, epoch, best_val_loss, checkpoint_dir, is_best_so_far):
#         # Make the checkpoint
#         checkpoint = {}
#         checkpoint['next_epoch'] = epoch + 1
#         checkpoint['best_val_loss'] = best_val_loss
#         checkpoint['model_state_dict'] = model.state_dict()
#         checkpoint['optimizer_state_dict'] = optimizer.state_dict()
#
#         # Save it as the most up to date checkpoint
#         filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
#         torch.save(checkpoint, filename)
#
#         # Save it as the "best" checkpoint if we are the best
#         if is_best_so_far:
#             best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
#             torch.save(checkpoint, best_filename)
#
#
#     def update_op(self, model, optimizer, minibatch, iter, args):
#         # compute the model output
#         output = model(minibatch)
#
#         # Compute the generator or discriminator loss, and make an appropriate step
#         regularization_loss = 0 # TODO
#         reconstruction_loss = 0 # TODO
#         total_loss = regularization_loss + reconstruction_loss
#
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#
#         losses["regularization_loss"] = regularization_loss
#         losses["reconstruction_loss"] = reconstruction_loss
#         losses["total_loss"] = total_loss
#
#         # We should put "losses" here that we can to compute per batch + not avg per epoch
#         losses["weight_norm"] = 0 # TODO
#         losses["update_magnitude"] = 0 # TODO
#         losses["update_ratio"] = 0 # TODO
#
#         # Return the dictionary of 'losses'
#         return losses
#
#
#     def validation_loss(self, model, minibatch):
#         # compute the model output
#         output = model(minibatch)
#
#         # Compute the generator or discriminator loss, and make an appropriate step
#         regularization_loss = 0 # TODO
#         reconstruction_loss = 0 # TODO
#         total_loss = regularization_loss + reconstruction_loss
#
#         losses["regularization_loss"] = regularization_loss
#         losses["reconstruction_loss"] = reconstruction_loss
#         losses["total_loss"] = total_loss
#
#         # Return the dictionary of losses
#         return losses