import cv2
import numpy as np

def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    #if blur:
    img_b = cv2.GaussianBlur(img_b,(13,13),0)
    img_b = cv2.bilateralFilter(img_b.astype(np.float32),7,85,85)
    img_a = cv2.GaussianBlur(img_a,(13,13),0)
    img_a = cv2.bilateralFilter(img_a.astype(np.float32),7,85,85)
    img_b = cv2.medianBlur(img_b,5)
    img_a = cv2.medianBlur(img_a,5)
    win = k_size
    #win=50#30#15#30 #15 for 1b in experiment.py **MAKE THIS AN INPUT**
    assert img_a.shape == img_b.shape
    I_x = np.zeros(img_a.shape)
    I_y = np.zeros(img_a.shape)
    I_t = np.zeros(img_a.shape)
    I_x[1:-1, 1:-1] = (img_a[1:-1, 2:] - img_a[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (img_a[2:, 1:-1] - img_a[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = img_a[1:-1, 1:-1] - img_b[1:-1, 1:-1]
    params = np.zeros(img_a.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (cum_params[2*win+1:,2*win+1:]-
                   cum_params[2*win+1:,:-1-2*win]-
                   cum_params[:-1-2*win,2*win+1:]+
                   cum_params[:-1-2*win,:-1-2*win])
    det = win_params[...,0]*win_params[...,1]-win_params[...,2]**2
    op_flow_u = np.zeros(img_a.shape)
    op_flow_v = np.zeros(img_a.shape)
    u = np.where(det != 0,
                 (win_params[...,1]*win_params[...,3]-
                  win_params[...,2]*win_params[...,4])/det,0)
    v = np.where(det != 0,
                 (win_params[...,0]*win_params[...,4]-
                  win_params[...,2]*win_params[...,3])/det,0)
    op_flow_u[win + 1: -1 - win, win + 1: -1 - win] = u[:-1, :-1]
    op_flow_v[win + 1: -1 - win, win + 1: -1 - win] = v[:-1, :-1]
    return (op_flow_u,op_flow_v)



shift_0 = cv2.imread('Shift0.png', 0) / 255.
shift_r2 = cv2.imread('ShiftR2.png', 0) / 255.
shift_r5_u5 = cv2.imread('ShiftR5U5.png', 0) / 255.
k_size = 50  # TODO: Select a kernel size
k_type = ""  # TODO: Select a kernel type
sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
u, v = optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)
u_v = quiver(u, v, scale=3, stride=10)
cv2.imwrite('right_shift.png', u_v)
u, v = optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)
u_v = quiver(u, v, scale=3, stride=10)
cv2.imwrite('up_right_shift.png', u_v)