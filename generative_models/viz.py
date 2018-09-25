import torch
import torch.nn as nn
import torch.nn.functional as F





def regress_latent(generative_model, x, latent_size, quiet=True):
    # Make a latent parameter that we can compute gradients for
    batch_size = x.size()[0]
    z_shape = torch.Size([batch_size, latent_size])
    z_init = torch.normal(torch.zeros(z_shape), torch.ones(z_shape))
    z = torch.autograd.Variable(z_init)
    z.requires_grad_()

    # Setup optimization, and make sure that the model is in eval mode
    lr = 0.001
    loss_fn = torch.nn.MSELoss(reduction='None')
    optimizer = torch.optim.Adam(z.parameters(), lr=lr, amsgrad=True)
    generative_model.eval()

    # Regress as far as we can
    x_gen = generative_model(z)
    best_z = z.clone()
    best_err = loss(x0, x_gen)

    for _ in range(4):
        i = 0
        while i < 1e4 and steps_since_last_improvement < 40:
            # counting num updates
            i += 1

            # Compute loss
            x_gen = generative_model(z)
            loss = torch.sum(loss_fn(x, x_gen), dim=1)

            # Update any best values found
            mask = loss < best_err
            best_z[mask] = z[mask]
            best_err[mask] = loss[mask]
            steps_since_last_improvement += 1
            if torch.sum(mask) > 0:
                steps_since_last_improvement = 0

            # Log some things
            if not quiet and i % 100 == 0:
                print("lr=%f, iter=%d, cur_err_avg=%f, best_err_avg=%f" % (lr, i, torch.mean(loss), torch.mean(best_err)))

            # Make a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # DEBUG: print what the break condition was
        if not quiet:
            print("Value of i is %d, and break condition is i < 1e4" % i)
            print("Value of steps_since_last_improve is $d and condition is _ < 40" % steps_since_last_improve)

        # Reduce the learning rate to see if can refine even further
        lr /= 4.0

    return best_z, best_err





def linear_interpolate(generative_model, x1, x2, latent_size, num_steps, quiet=True):
    """
    Given a generative model and two data samples x1, x2 with shape (D,), produce a the closes linear interpolation
    between the two datapoints as can be produced by the generative model. Specifically, we need to find a sequence
    y1, y2, ..., yk, where k is 'num_steps', where y1 and yk are the generators "nearest neighbours" to x1 and x2.
    If y1 is generated from latent z1 and yk from zk. Then let z2, ..., z(k-1), be linear interpolations between z1 and
    zk. y2, ..., y(k-1) are then the generated samples from z2, ..., z(k-1). This function will return y1, ..., yk.

    :param generative_model: The generative model to use for the linear interpolation
    :param x1: An example, from real data, to be used as an endpoint in the linear interpolations.
    :param x2: Also an example, from real data, to be used as an endpoint in the linear interpolations.
    :param latent_size: The size of the latent variables used by the generative model.
    :param num_steps: The number of steps to take in the linear intepolation (in the latent space).
    :param quiet: Boolean for if we should
    :return: An interpolation (linear in the latent space) produced using the generative model.
    """
    # Make sure running in eval mode
    generative_model.eval()

    # Regress for the endpoints
    z1 = regress_latent(generative_model, x1, latent_size, quiet)
    z2 = regress_latent(generative_model, x2, latent_size, quiet)

    # Produce the linear interpolations step by step
    z = z1
    delta = (z2 - z1) / (num_steps+1)
    interpolations = []
    interpolations.append(x1_gen)
    for _ in range(num_steps+2):
        x = generative_model(z)
        interpolations.append(x)
        z += delta

    return interpolations