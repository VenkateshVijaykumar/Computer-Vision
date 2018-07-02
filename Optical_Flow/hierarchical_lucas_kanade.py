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

    #raise NotImplementedError

def custom_kernel(val):
    w_1d = np.array([0.25 - val/2.0, 0.25, val, 0.25, 0.25 - val/2.0])
    return np.outer(w_1d, w_1d)

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    out = None
    kernel = custom_kernel(0.375)
    outimage = cv2.filter2D(image,-1,kernel)
    #outimage = scipy.signal.convolve2d(image,kernel,'same')
    out = outimage[::2,::2]
    return out

    #raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    output = []
    output.append(image)
    tmp = image
    for i in range(0,levels-1):
      tmp = reduce_image(tmp)
      output.append(tmp)
    return output 

def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    out = None
    kernel = custom_kernel(0.375)
    outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
    outimage[::2,::2]=image[:,:]
    out = 4*cv2.filter2D(outimage,-1,kernel) #scipy.signal.convolve2d(outimage,kernel,'same')
    return out

def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    h,w = image.shape[:2]
    x,y = np.meshgrid(xrange(w),xrange(h))
    x,y = cv2.convertMaps(x.astype(np.float32),y.astype(np.float32), cv2.CV_16SC2)
    mapU_unscaled = U
    mapV_unscaled = V
    mapU, mapV = cv2.convertMaps(mapU_unscaled.astype(np.float32),mapV_unscaled.astype(np.float32), cv2.CV_16SC2)
    mapX = mapU + x
    mapY = mapV + y
    #mapX = np.zeros(image.shape[:2],np.float32)
    #mapY = np.zeros(image.shape[:2],np.float32)
    h,w = image.shape[:2]
    #for j in range(h):
    #  for i in range(w):
    #    mapX.itemset((j,i),i+U[j,i])
    #    mapY.itemset((j,i),j+V[j,i])
    warped_img = cv2.remap(image, mapY, mapX, interpolation=interpolation, borderMode=border_mode)
    return warped_img

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    k_size = k_size  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0
    gpyr_a = gaussian_pyramid(img_a, levels)
    gpyr_b = gaussian_pyramid(img_b, levels)
    u, v = optic_flow_lk(gpyr_a[-1], gpyr_b[-1], k_size, k_type, sigma)
    exu =  expand_image(u)
    exv = expand_image(v)
    if levels > 1:
      levels -=1
    else:
      levels = levels
    for i in xrange(levels,0,-1):
      ul,vl = optic_flow_lk(gpyr_a[i], gpyr_b[i], k_size, k_type, sigma)
      exul  = expand_image(ul)
      exvl  = expand_image(vl)
      exul  = np.multiply(exul,2.0)
      exvl  = np.multiply(exvl,2.0)
      interpolation = cv2.INTER_CUBIC  # You may try different values
      border_mode = cv2.BORDER_REFLECT101  # You may try different values
      warped_im = warp(gpyr_a[i-1], exul, exvl, interpolation, border_mode)
      ucl,vcl = optic_flow_lk(warped_im, gpyr_b[i-1], k_size, k_type, sigma)
      exul += ucl
      exvl += vcl
    final_u = exul
    final_v = exvl
    return final_u,final_v

urban_img_01 = cv2.imread('urban01.png', 0) / 255.
urban_img_02 = cv2.imread('urban02.png', 0) / 255.
levels = 3  # TODO: Define the number of levels
k_size = 50  # TODO: Select a kernel size
k_type = ""  # TODO: Select a kernel type
sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
interpolation = cv2.INTER_CUBIC  # You may try different values
border_mode = cv2.BORDER_REFLECT101  # You may try different values
u, v = hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                           k_type, sigma, interpolation, border_mode)
u_v = quiver(u, v, scale=3, stride=10)
cv2.imwrite('hierarchical_shift.png', u_v)