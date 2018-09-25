############
# Not Used #
############


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# from __future__ import division
#
# import numpy as np
# import torch
#
# # Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
# H36M_NAMES = ['']*32
# H36M_NAMES[0]  = 'Hip'
# H36M_NAMES[1]  = 'RHip'
# H36M_NAMES[2]  = 'RKnee'
# H36M_NAMES[3]  = 'RFoot'
# H36M_NAMES[6]  = 'LHip'
# H36M_NAMES[7]  = 'LKnee'
# H36M_NAMES[8]  = 'LFoot'
# H36M_NAMES[12] = 'Spine'
# H36M_NAMES[13] = 'Thorax'
# H36M_NAMES[14] = 'Neck/Nose'
# H36M_NAMES[15] = 'Head'
# H36M_NAMES[17] = 'LShoulder'
# H36M_NAMES[18] = 'LElbow'
# H36M_NAMES[19] = 'LWrist'
# H36M_NAMES[25] = 'RShoulder'
# H36M_NAMES[26] = 'RElbow'
# H36M_NAMES[27] = 'RWrist'
#
#
# dims_to_use = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]
# indx_to_use_3d = [ 0 * 3,  0 * 3 + 1,  0 * 3 + 2,
#                    1 * 3,  1 * 3 + 1,  1 * 3 + 2,
#                    2 * 3,  2 * 3 + 1,  2 * 3 + 2,
#                    3 * 3,  3 * 3 + 1,  3 * 3 + 2,
#                    6 * 3,  6 * 3 + 1,  6 * 3 + 2,
#                    7 * 3,  7 * 3 + 1,  7 * 3 + 2,
#                    8 * 3,  8 * 3 + 1,  8 * 3 + 2,
#                   12 * 3, 12 * 3 + 1, 12 * 3 + 2,
#                   13 * 3, 13 * 3 + 1, 13 * 3 + 2,
#                   15 * 3, 15 * 3 + 1, 15 * 3 + 2,
#                   17 * 3, 17 * 3 + 1, 17 * 3 + 2,
#                   18 * 3, 18 * 3 + 1, 18 * 3 + 2,
#                   19 * 3, 19 * 3 + 1, 19 * 3 + 2,
#                   25 * 3, 25 * 3 + 1, 25 * 3 + 2,
#                   26 * 3, 26 * 3 + 1, 26 * 3 + 2,
#                   27 * 3, 27 * 3 + 1, 27 * 3 + 2]
# indx_to_use_2d = [ 0 * 2,  0 * 2 + 1,
#                    1 * 2,  1 * 2 + 1,
#                    2 * 2,  2 * 2 + 1,
#                    3 * 2,  3 * 2 + 1,
#                    6 * 2,  6 * 2 + 1,
#                    7 * 2,  7 * 2 + 1,
#                    8 * 2,  8 * 2 + 1,
#                   12 * 2, 12 * 2 + 1,
#                   13 * 2, 13 * 2 + 1,
#                   15 * 2, 15 * 2 + 1,
#                   17 * 2, 17 * 2 + 1,
#                   18 * 2, 18 * 2 + 1,
#                   19 * 2, 19 * 2 + 1,
#                   25 * 2, 25 * 2 + 1,
#                   26 * 2, 26 * 2 + 1,
#                   27 * 2, 27 * 2 + 1]
#
#
#
# def unNormalize3DData(normalized_data, data_mean, data_std):
#     data_mean = data_mean[indx_to_use_3d]
#     data_std = data_std[indx_to_use_3d]
#     return visUnNormalizeData(normalized_data, data_mean, data_std)
#
# def unNormalize2DData(normalized_data, data_mean, data_std):
#     data_mean = data_mean[indx_to_use_2d]
#     data_std = data_std[indx_to_use_2d]
#     return visUnNormalizeData(normalized_data, data_mean, data_std)
#
# def visUnNormalizeData(normalized_data, data_mean, data_std):
#     if normalized_data.ndim == 1:
#         normalized_data = np.expand_dims(normalized_data, 0)
#
#     T = normalized_data.shape[0]  # Batch size
#     D = data_mean.shape[0]  # 32 for 2d, 48 for 3d
#
#     orig_data = np.zeros((T, D), dtype=np.float32)
#     orig_data += normalized_data
#
#     # Multiply times stdev and add the mean
#     stdMat = data_std.reshape((1, D))
#     stdMat = np.repeat(stdMat, T, axis=0)
#     meanMat = data_mean.reshape((1, D))
#     meanMat = np.repeat(meanMat, T, axis=0)
#     orig_data = np.multiply(orig_data, stdMat) + meanMat
#     return orig_data
#
# def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
#     T = normalized_data.shape[0]  # Batch size
#     D = data_mean.shape[0]  # 96
#
#     orig_data = np.zeros((T, D), dtype=np.float32)
#
#     orig_data[:, dimensions_to_use] = normalized_data
#
#     # Multiply times stdev and add the mean
#     stdMat = data_std.reshape((1, D))
#     stdMat = np.repeat(stdMat, T, axis=0)
#     meanMat = data_mean.reshape((1, D))
#     meanMat = np.repeat(meanMat, T, axis=0)
#     orig_data = np.multiply(orig_data, stdMat) + meanMat
#     return orig_data
#
# def unNormalizeDataTorch(normalized_data, data_mean, data_std, dimensions_to_use, cuda=True):
#     T = normalized_data.size(0)  # Batch size
#     D = data_mean.shape[0]  # 96
#
#     orig_data = torch.autograd.Variable(torch.zeros((T, D)))
#     if cuda: orig_data = orig_data.cuda()
#     orig_data[:, dimensions_to_use] = normalized_data
#
#     # Multiply times stdev and add the mean
#     stdMat = data_std.reshape((1, D))
#     stdMat = torch.Tensor(np.repeat(stdMat, T, axis=0))
#     meanMat = data_mean.reshape((1, D))
#     meanMat = torch.Tensor(np.repeat(meanMat, T, axis=0))
#     if cuda:
#         stdMat = stdMat.cuda()
#         meanMat = meanMat.cuda()
#     orig_data = torch.mul(orig_data, stdMat) + meanMat
#     return orig_data[:, dimensions_to_use] # this is to be used WITH a network => only keep the dimensions we want around
#
#
# def reNormalizeDataTorch(unormalized_data, data_mean, data_std, dimensions_to_use, cuda=True):
#     T = unormalized_data.size(0)  # Batch size
#     D = data_mean.shape[0]  # 96
#
#     new_data = torch.autograd.Variable(torch.zeros((T, D)))
#     if cuda: new_data = new_data.cuda()
#     new_data[:, dimensions_to_use] = unormalized_data
#
#     # Multiply times stdev and add the mean
#     stdMat = data_std.reshape((1, D))
#     stdMat = torch.Tensor(np.repeat(stdMat, T, axis=0))
#     meanMat = data_mean.reshape((1, D))
#     meanMat = torch.Tensor(np.repeat(meanMat, T, axis=0))
#     if cuda:
#         stdMat = stdMat.cuda()
#         meanMat = meanMat.cuda()
#     norm_data = torch.div(new_data - meanMat, stdMat)
#     return norm_data[:, dimensions_to_use] # this is to be used WITH a network => only keep the dimensions we want around
#
