
# constant options for how to draw the pose
pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]      # parent relations between joints
linewidth = 8                                                   # line width for skeleton
colors = [[255, 0, 0],      [255, 85, 0],       [255, 170, 0],  # colors to use for skeleton
          [255, 255, 0],    [170, 255, 0],      [85, 255, 0],
          [0, 255, 0],      [0, 255, 85],       [0, 255, 170],
          [0, 255, 255],    [0, 170, 255],      [0, 85, 255],
          [0, 0, 255],      [85, 0, 255],       [170,0,255],
          [255,0,255]]

def visualize(oriImg, points):
    import matplotlib
    import cv2 as cv
    import matplotlib.pyplot as plt
    import math

    canvas = oriImg
    x = points[:, 0]
    y = points[:, 1]

    for n in range(len(x)):
        for child in range(len(pa)):
            if pa[child] is 0:
                continue

            x1 = x[pa[child] - 1]
            y1 = y[pa[child] - 1]
            x2 = x[child]
            y2 = y[child]

            cv.line(canvas, (x1, y1), (x2, y2), colors[child], linewidth)

    # return the original image with the overlay
    return canvas[:, :, [2, 1, 0]]