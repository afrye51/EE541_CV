import skimage
from skimage import io
from skimage import data
from skimage.transform import resize
from skimage.transform import rotate
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA
import time
from scipy import signal
from scipy import linalg
from scipy.ndimage import gaussian_filter
import cv2


################### LAB 1 FUNCTIONS ##################


def convolv_2d_grayscale(image, kernel, mode='constant', boundary='0'):
    
    image = skimage.img_as_float(image)
    
    im_shape = np.shape(image)
    im_x = im_shape[0]
    im_y = im_shape[1]
    
    kernel = np.fliplr(np.flipud(kernel))
    kernel_shape = np.shape(kernel)
    k_x = int((kernel_shape[0] - 1) / 2)
    k_y = int((kernel_shape[1] - 1) / 2)
    new_image = np.zeros(im_shape)
    
    if mode == 'constant':
        image = np.lib.pad(image, (k_x, k_y), 'constant', constant_values=(boundary, boundary))
    elif mode == 'edge':
        image = np.lib.pad(image, (k_x, k_y), 'edge')
   
    for i in range(im_x):
        for j in range(im_y):
            new_image[i, j] = np.max([0, np.min([1, np.sum(image[i : i + 2*k_x + 1, j : j + 2*k_y + 1] * kernel)])])
    
    return new_image


def my_imfilter(image, kernel, mode='constant', boundary='0'):
    kernel_shape = np.shape(kernel)
    im_shape = np.shape(image)
    k_x = kernel_shape[0] - 1
    k_y = kernel_shape[1] - 1
    
    if len(kernel_shape) != 2 or k_x % 2 != 0 or k_y % 2 != 0:
        print("Please enter an odd-dimension kernel")
        return None
    
    if mode != 'constant' and mode != 'edge':
        print("Invalid mode")
        return None

    if len(im_shape) == 2:
        # Grayscale image
        #print('grayscale image')
        image = convolv_2d_grayscale(image, kernel, mode, boundary)
    elif len(im_shape) == 3:
        # RGB image
        #print('rgb image')
        for i in range(im_shape[2]):
            image[:,:,i] = convolv_2d_grayscale(image[:,:,i], kernel, mode, boundary)
    else:
        print('Image error')
        return None
    
    return image


def gaussian_2d(stdev, shape):
    
    side_x = int((shape[0] - 1) / 2)
    side_y = int((shape[1] - 1) / 2)
    x = np.arange(-side_x, side_x + 1, 1, dtype=float)[None]
    y = np.arange(-side_y, side_y + 1, 1, dtype=float)[None]
    x = 1 / (stdev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / stdev) ** 2)
    y = 1 / (stdev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (y / stdev) ** 2)
    temp = x.T @ y
    temp /= np.sum(temp)
    return temp
    

# Assumes both images are either grayscale or RGB, not one of each
def make_images_same_size(images):
    num_im = np.shape(images)
    if len(num_im) == 2:
        return images
    num_im = num_im[0]
    im_shapes = []
    for image in images: 
        im_shapes.append(np.shape(image))
    im_shapes = np.array(im_shapes)
    im_shapes_split = np.hsplit(im_shapes, 2)
    x = np.max(im_shapes_split[0])
    y = np.max(im_shapes_split[1])
    result = np.zeros((num_im, x, y))
    for i in range(num_im): 
        result[i] = resize(images[i], (x, y))
    return result
    
    
def hybrid_fourier_grayscale(image1, image2, stdev=24, delta=8, show=False):
    
    shape = np.shape(image1)
    
    fft1 = np.fft.fft2(image1)
    fft1 = np.fft.fftshift(fft1)
    fft2 = np.fft.fft2(image2)
    fft2 = np.fft.fftshift(fft2)
    
    filter1 = gaussian_2d(stdev, shape)
    mx = np.max(filter1)
    filter1 = 1/mx * filter1
    
    stdev2 = stdev + delta
    filter2 = gaussian_2d(stdev2, shape)
    mx = np.max(filter2)
    filter2 = 2 - (filter2 / mx)
    
    fft1_f = fft1 * filter1
    fft2_f = fft2 * filter2
    
    fft1_fs = np.fft.ifftshift(fft1_f)
    image1_f = np.real(np.fft.ifft2(fft1_fs))
    fft2_fs = np.fft.ifftshift(fft2_f)
    image2_f = np.real(np.fft.ifft2(fft2_fs))
    
    hybrid_image = 0.5 * (image1_f + image2_f)
    
    if show:
        plt.figure(figsize=(15, 20))
        plt.subplot(421)
        plt.imshow(image1, cmap='gray')
        plt.axis('off')
        plt.title('Image 1')
        plt.subplot(422)
        plt.imshow(image2, cmap='gray')
        plt.axis('off')
        plt.title('Image 2')
        plt.subplot(423)
        plt.imshow(20*np.log10(np.abs(fft1)), cmap='gray')
        plt.axis('off')
        plt.title('Image 1 fft')
        plt.subplot(424)
        plt.imshow(20*np.log10(np.abs(fft2)), cmap='gray')
        plt.axis('off')
        plt.title('Image 2 fft')
        plt.subplot(425)
        plt.imshow(20*np.log10(np.abs(filter1)), cmap='gray')
        plt.axis('off')
        plt.title('Image 1 filter')
        plt.subplot(426)
        plt.imshow(20*np.log10(np.abs(filter2)), cmap='gray')
        plt.axis('off')
        plt.title('Image 2 filter')
        plt.subplot(427)
        plt.imshow(20*np.log10(np.abs(fft1_f)), cmap='gray')
        plt.axis('off')
        plt.title('Image 1 fft filtered')
        plt.subplot(428)
        plt.imshow(20*np.log10(np.abs(fft2_f)), cmap='gray')
        plt.axis('off')
        plt.title('Image 2 fft filtered')
        
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.imshow(image1_f, cmap='gray')
        plt.axis('off')
        plt.title('Image 1 filtered')
        plt.subplot(222)
        plt.imshow(image2_f, cmap='gray')
        plt.axis('off')
        plt.title('Image 2 filtered')
        plt.subplot(223)
        plt.imshow(hybrid_image, cmap='gray')
        plt.axis('off')
        plt.title('Hybrid image')
    
    return hybrid_image


def hybrid_fourier(image1, image2, stdev=24, delta=8, show=False):
    
    image1 = skimage.img_as_float(image1)
    image2 = skimage.img_as_float(image2)
    
    im1_shape = np.shape(image1)
    im2_shape = np.shape(image2)
    if im1_shape != im2_shape:
        print("Please enter images of the same size")
        return None
    
    if len(im1_shape) == 2:
        # Grayscale image
        #print('grayscale image')
        image1 = hybrid_fourier_grayscale(image1, image2, stdev, delta, show)
    elif len(im1_shape) == 3:
        # RGB image
        #print('rgb image')
        for i in range(im1_shape[2]):
            image1[:,:,i] = hybrid_fourier_grayscale(image1[:,:,i], image2[:,:,i], stdev, delta, show)
    else:
        print('Image error')
        return None
    
    return image1


def hybrid_convolution(image1, image2, stdev=24, delta=8):
    
    shape = [stdev * 3, stdev * 3]
    filter1 = gaussian_2d(stdev, shape)
    stdev2 = stdev + delta
    shape = [stdev2 * 3, stdev2 * 3]
    filter2 = -gaussian_2d(stdev2, shape)
    filter2[int(shape[0] / 2),int(shape[1] / 2)] = -2 * np.sum(filter2) + filter2[int(shape[0] / 2),int(shape[1] / 2)]
    image1 = my_imfilter(image1, filter1, 'edge')
    image2 = my_imfilter(image2, filter2, 'edge')
    return 0.5 * (image1 + image2)


def hybrid_image(image1, image2, stdev=24, delta=8, method='fourier'):
    
    image1 = skimage.img_as_float(image1)
    image2 = skimage.img_as_float(image2)
    
    image1, image2, ret = make_images_same_size(image1, image2)
    if not ret:
        print('Invalid Images, check dimensions')
        return None
    
    if method == 'fourier':
        return hybrid_fourier(image1, image2, stdev, delta)
    elif method == 'convolution':
        return hybrid_convolution(image1, image2, stdev, delta)
    else:
        print("Invalid method")
        return None


def plot_kernel_image_time(im_steps, k_min, k_max, k_step, mode='scipy', n_measurements=5):
    
    if k_min % 2 == 0:
        k_min += 1
    if k_step % 2 != 0:
        k_step += 1
    
    #im_steps = np.arange(im_min, im_max, im_step)
    k_steps = np.arange(k_min, k_max, k_step)
    
    results = np.zeros((len(im_steps), len(k_steps)))
    for useless in range(n_measurements):
        pct = useless / n_measurements * 100
        print(str(pct) + '% completed')
        start2 = time.time()
        for i in range(len(im_steps)):
            for j in range(len(k_steps)):
                im = np.full((im_steps[i], im_steps[i]), 0.5)
                k = np.full((k_steps[j], k_steps[j]), 0.5)
                start = time.time()
                if mode == 'custom':
                    im = convolv_2d_grayscale(im, k)
                elif mode == 'scipy':
                    im = signal.convolve2d(im, k)
                else:
                    print('Invalid mode')
                    return
                results[i,j] += (time.time() - start) / n_measurements
        delta2 = time.time() - start2
        print('Estimated time remaining: ' + str(delta2 * (n_measurements - useless - 1)))
    
    x, y = np.meshgrid(im_steps, k_steps)
    print(x)
    print(y)
    print(results)
    fig = plt.figure()
    fig.set_size_inches(11,8)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(y.T, x.T, results)

    # Customize the z axis.
    ax.set_zlim(0, np.max(results))
    ax.set_xlabel('Kernel size (pixels)')
    ax.set_ylabel('Image size (pixels)')
    ax.set_zlabel('Computation Time (seconds)')


def image_pyramid(image, width_ratios=[8, 4, 2, 1]):
    temp = image
    fig = plt.figure()
    fig.set_size_inches(11,8)
    n = len(width_ratios)
    gs = gridspec.GridSpec(1, n, width_ratios=width_ratios, wspace=0)
    for i in range(n):
        plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(temp)
        shape = np.shape(temp)
        temp = resize(temp, (int(shape[0]/2), int(shape[1]/2)))


def ajacent_images(images):
    images = make_images_same_size(images)
    shape = np.shape(images)
    fig = plt.figure()
    fig.set_size_inches(11,8)
    if len(shape) == 4 or len(shape) == 3:
        n = shape[0]
    elif len(shape) == 2: 
        n = 1
    else:
        print('Image Error')
        return
    ratios = np.ones(n)
    gs = gridspec.GridSpec(1, n, width_ratios=ratios, wspace=0)
    for i in range(n):
        plt.subplot(gs[i])
        plt.axis('off')
        if len(shape) == 4: 
            plt.imshow(images[i])
        elif len(shape) == 3: 
            plt.imshow(images[i], cmap='gray')
        elif len(shape) == 2: 
            plt.imshow(images, cmap='gray')
        else:
            print('Image error')
            return
    return n, shape[0], shape[1]


############### LAB 2 FUNCTIONS ###################


## Expects:
##  loc: location (x, y) of the feature (in pixels)
##  mag: size to display the feature at (side length pf square in pizels)
##  t: orientation of the feature measured from the +x axis (radians)
def showFeatures(loc, mag, t):
    l2 = int(mag / 2)
    square = np.array([[0, 0], [l2, 0], [l2, l2], [-l2, l2], [-l2, -l2], [l2, -l2], [l2, 0]])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    square = square @ R
    x, y = np.hsplit(square, 2)
    plt.plot(x + loc[0], y + loc[1], 'r-')


def gaussian_2d_order(size, stdev, order=1):
    identity = np.zeros((size, size))
    identity[size // 2, size // 2] = 1
    kernel = gaussian_filter(identity, sigma=stdev, order=order)
    return kernel


def plot_gaussian_2d(gaussian):
    shape = np.shape(gaussian)
    x = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    y = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    fig.set_size_inches(11,8)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(y.T, x.T, gaussian)


def harris_alpha(i_xx2, i_yy2, i_xy2, alpha=0.04):
    result = i_xx2*i_yy2 - i_xy2**2 - alpha * (i_xx2 + i_yy2)**2
    return result / np.max(result)


def harris_eig(i_xx2, i_yy2, i_xy2):
    det = i_xx2*i_yy2 - i_xy2**2
    trace = i_xx2 + i_yy2
    result = (det / trace)
    return result / np.max(result)


def detect_corners_harris(image, s1=1, s2=2, mode='eig'):
    #k = gaussian_2d_order(10, s1, (0,1))
    #k2 = gaussian_2d_order(10, s2, 0)
    #i_x = signal.convolve2d(image, k, mode='same')
    #i_y = signal.convolve2d(image, k.T, mode='same')
    i_x = gaussian_filter(image, sigma=s1, order=(0,1))
    i_y = gaussian_filter(image, sigma=s1, order=(1,0))
    i_xy = i_x*i_y
    i_xx = i_x**2
    i_yy = i_y**2
    #i_xy2 = signal.convolve2d(i_xy, k2, mode='same')
    #i_xx2 = signal.convolve2d(i_xx, k2, mode='same')
    #i_yy2 = signal.convolve2d(i_yy, k2, mode='same')
    i_xy2 = gaussian_filter(i_xy, sigma=s2, order=0)
    i_xx2 = gaussian_filter(i_xx, sigma=s2, order=0)
    i_yy2 = gaussian_filter(i_yy, sigma=s2, order=0)
    if mode == 'eig':
        result = harris_eig(i_xx2, i_yy2, i_xy2)
    elif mode == 'alpha':
        result = harris_alpha(i_xx2, i_yy2, i_xy2)
    else:
        print('Invalid Mode')
        return None
    theta = np.arctan2(i_y, i_x)
    return result, theta


def threshold_image(image, threshold=0.5):
    im_np = np.array(image)
    im_np[im_np < threshold] = 0
    #im_np[im_np >= threshold] = 1
    return im_np


def local_maxima_dumb(image, n=5):
    # for each pixel (% n) in the image
        # get (x, y) of local maxima
        # set all other pixels in area to 0
        # set that pixel to 1
    im = np.copy(np.array(image))
    shape = np.shape(im)
    print(shape)
    for i in range(shape[0] // n + 1):
        for j in range(shape[1] // n + 1):
            x0 = n * i
            x1 = x0 + n
            y0 = n * j
            y1 = y0 + n
            if x1 > shape[0]:
                x1 = shape[0]
            if y1 > shape[1]:
                y1 = shape[1]
            if x1 != x0 and y1 != y0:
                ind = np.unravel_index(np.argmax(im[x0:x1, y0:y1], axis=None), (x1-x0, y1-y0))
                ind = np.add(ind, (x0, y0))
                mx = im[ind[0], ind[1]]
                im[x0:x1, y0:y1] = 0
                im[ind[0], ind[1]] = mx
    return im


def local_maxima_descent(image, pix_dist=3, threshold=0.3):
    # for every pixel in the image:
        # If an ajacent pixel is larger, move to it
        # else store that pixel's (x, y, mag) in a dictionary type thing
    direc = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    im_shape = np.shape(image)
    local_max = []
    current = []
    nxt = []
    for i in range(im_shape[0] // pix_dist):
        for j in range(im_shape[1] // pix_dist):
            i_im = i * pix_dist
            j_im = j * pix_dist
            nxt.append([i_im, j_im])
    while len(nxt) > 0:
        current = np.copy(nxt)
        nxt = []
        for point in current:
            if image[point[0], point[1]] > threshold:
                found = False
                for d in direc:
                    tmp = [point[0] + d[0], point[1] + d[1]]
                    if not found and (tmp[0] < im_shape[0]) and (tmp[1] < im_shape[1]):
                        if image[tmp[0], tmp[1]] > image[point[0], point[1]]:
                            found = True
                            nxt.append(tmp)
                if not found:
                    local_max.append(point)
    result = np.unique(local_max, axis=0)
    im_filt = np.zeros(np.shape(image))
    index = np.split(result, 2, axis=1)
    im_filt[tuple(index)] = 1
    im_filt2 = im_filt * image
    return im_filt, im_filt2, result


def plot_boxes_im(im, s1=1, s2=2, pix_dist=3, threshold=0.3):
    check, theta = detect_corners_harris(im, s1, s2)
    check_filt, check_filt2, result = local_maxima_descent(check, pix_dist, threshold)
    
    index = np.split(result, 2, axis=1)
    angles = theta[tuple(index)]
    mag = check_filt2[tuple(index)]
    ajacent_images(im)

    for i in range(len(angles)):
        showFeatures([result[i][1], result[i][0]], 50 * mag[i][0], theta[i][0])


def plot_boxes_desc(feats, image=None):
    if image is not None:
        ajacent_images(image)
    n = np.shape(feats)
    for i in range(n[0]):
        loc, theta, mag, vect = feats[i]
        showFeatures([loc[1], loc[0]], 50 * mag[0], theta[0])


def grab_box(image, loc, size=5):
    ims = np.shape(image)
    n = size // 2
    x_min = loc[0] - n
    x_max = loc[0] + n + 1
    y_min = loc[1] - n
    y_max = loc[1] + n + 1
    if x_min < 0 or y_min < 0 or x_max > ims[0] or y_max > ims[1]:
        return None
    else:
        return image[x_min:x_max, y_min:y_max]


def features_descriptors(image, s1=1, s2=2, pix_dist=3, threshold=0.3):
    feat, theta = detect_corners_harris(image, s1, s2)
    feat_filt_max, feat_filt, loc = local_maxima_descent(feat, pix_dist, threshold)
    index = np.split(loc, 2, axis=1)
    angles = theta[tuple(index)]
    mag = feat_filt[tuple(index)]
    return np.array(create_descriptors(image, loc, angles, mag))


def create_descriptors(image, locs, thetas, mags):
    descriptors = []
    for i in range(len(thetas)):
        desc = create_descriptor(image, locs[i], thetas[i], mags[i])
        if desc is not None:
            descriptors.append(desc)
    return descriptors


def create_descriptor(image, loc, theta, mag):
    im_rot = rotate_and_resize(image, loc, theta, 5)
    if im_rot is None:
        return None
    vect = np.reshape(im_rot, (25, 1))
    vect = vect / LA.norm(vect)
    return [loc, theta, mag, vect]


def rotate_and_resize(image, loc, theta, ret_size):
    im_rot = grab_box(image, loc, size=(2 * ret_size + 1))
    if im_rot is None:
        return None
    im_rot = rotate(im_rot, theta * 180 / np.pi)
    return grab_box(im_rot, [ret_size, ret_size], ret_size)


def diff_features(f1, f2):
    temp, temp, temp, vect1 = f1
    temp, temp, temp, vect2 = f2
    return LA.norm(vect2 - vect1)


def compare_features_old(f1s, f2s, threshold=0.2):
    f1_shape = np.shape(f1s)[0]
    f2_shape = np.shape(f2s)[0]
    results = np.zeros((f1_shape, f2_shape))
    for i in range(f1_shape):
        for j in range(f2_shape):
            results[i, j] = diff_features(f1s[i], f2s[j])
            
    diff = np.min(results, axis=0)
    match = []
    for i in range(f1_shape):
        if diff[i] < threshold:
            match.append([f1s[i], f2s[np.argmax(results[i], axis=None)]])#, diff[i]])
    return match


def compare_features_threshold(f1s, f2s, threshold=0.2):
    f1_shape = np.shape(f1s)[0]
    f2_shape = np.shape(f2s)[0]
    results = np.zeros((f1_shape, f2_shape))
    count = f1_shape // 10
    for i in range(f1_shape):
        for j in range(f2_shape):
            results[i, j] = diff_features(f1s[i], f2s[j])
        if i % count == 0:
            print(str(i // 10) + " out of " + str(count))
    match = []
    for i in range(np.min([f1_shape, f2_shape])):
        mx = np.unravel_index(np.argmin(results, axis=None), results.shape)
        #print(results[mx])
        if results[mx] < threshold:
            match.append([np.copy(f1s[mx[0]]), np.copy(f2s[mx[1]]), results[mx]])
        else:
            return match
        results[mx[0],:] = 10
        results[:,mx[1]] = 10
    return match


def compare_features_ratio(f1s, f2s, threshold=1):
    f1_shape = np.shape(f1s)[0]
    f2_shape = np.shape(f2s)[0]
    results = np.zeros((f1_shape, f2_shape))
    for i in range(f1_shape):
        for j in range(f2_shape):
            results[i, j] = diff_features(f1s[i], f2s[j])

    match = []
    for i in range(np.min([f1_shape, f2_shape])):
        mx = np.unravel_index(np.argmin(results, axis=None), results.shape)
        mx_val = results[mx]
        results[mx] = 10
        mx2_val = np.min(results[mx[0],:])
        ratio = mx2_val / mx_val
        if ratio > threshold:
            match.append([np.copy(f1s[mx[0]]), np.copy(f2s[mx[1]]), ratio])
        results[mx[0],:] = 10
        results[:,mx[1]] = 10
    return match


def connect_features(feats, im1, im2):
    im = np.concatenate((im1, im2), axis=1)
    ajacent_images(im)
    x = np.shape(im1[1])
    for f in feats:
        f[1][0][1] += x
        loc0, theta0, mag0, vect0 = f[0]
        loc1, theta1, mag1, vect1 = f[1]
        plot_boxes_desc([f[0]])
        plot_boxes_desc([f[1]])
        plt.plot([loc0[1], loc1[1]], [loc0[0], loc1[0]], 'b-')


################# LAB 3 FUNCTIONS #######################


def un_bias_image_sets(im_array):
    num_sets = np.shape(im_array)[0]
    num_im = np.shape(im_array)[1]
    im_array_unbiased = np.zeros(np.shape(im_array))
    for i in range(num_sets):
        im_norm = np.zeros((np.shape(im_array)[2], np.shape(im_array)[3]))
        for j in range(num_im):
            im_norm += im_array[i][j] / num_im
        for j in range(num_im):
            im_array_unbiased[i][j] = im_array[i][j] - im_norm
    return im_array_unbiased


def x_from_image_sets(images):
    num_sets = np.shape(images)[0]
    num_im = np.shape(images)[1]
    dim0 = np.shape(images)[2]
    dim1 = np.shape(images)[3]
    x = np.zeros((num_sets,dim0*dim1,num_im), dtype=np.float32)
    for i in range(num_sets):
        for j in range(num_im):
            x[i][:,j] = images[i][j].reshape(dim0*dim1)
    return x


def svd_from_x(x):
    num_sets = np.shape(x)[0]
    num_pix = np.shape(x)[1]
    num_im = np.shape(x)[2]
    u = np.zeros((num_sets,num_pix,num_im), dtype=np.float32)
    sv = np.zeros((num_sets,num_im), dtype=np.float32)
    vt = np.zeros((num_sets,num_im,num_im), dtype=np.float32)
    for i in range(num_sets):
        u[i], sv[i], vt[i] = np.linalg.svd(x[i], full_matrices=False)
    return np.array(u), np.array(sv), np.array(vt)


def m_from_u_x(u, x, k=None):
    if k is None:
        k = np.shape(x)[2]
    
    num_sets = np.shape(x)[0]
    num_im = np.shape(x)[2]
    m = np.zeros((num_sets,k,num_im))
    for i in range(num_sets):
        m[i] = u[i,:,0:k].T @ x[i]
    return m


def x_svd_m_from_image_sets(images):
    
    num_sets = np.shape(images)[0]
    num_im = np.shape(images)[1]
    dim0 = np.shape(images)[2]
    dim1 = np.shape(images)[3]
    x = np.zeros((num_sets,dim0*dim1,num_im), dtype=np.float32)
    u = np.zeros((num_sets,dim0*dim1,num_im), dtype=np.float32)
    sv = np.zeros((num_sets,num_im), dtype=np.float32)
    vt = np.zeros((num_sets,num_im,num_im), dtype=np.float32)
    m = np.zeros((num_sets,num_im,num_im))
    for i in range(num_sets):
        for j in range(num_im):
            x[i][:,j] = images[i][j].reshape(dim0*dim1)
        u[i], sv[i], vt[i] = np.linalg.svd(x[i], full_matrices=False)
        m[i] = u[i].T @ x[i]
    return x, u, sv, vt, m


def g_svd_m_from_x(x):
    
    num_sets = np.shape(x)[0]
    num_pix = np.shape(x)[1]
    num_im = np.shape(x)[2]
    
    g = np.zeros((num_pix,num_sets*num_im), dtype=np.float32)
    for i in range(num_sets):
        for j in range(num_im):
            temp = i*num_im + j
            g[:,temp] = x[i,:,j]
    u, sv, vt = np.linalg.svd(g, full_matrices=False)
    m = u.T @ g
    return g, u, sv, vt, m


def plot_3d_manifold(m, im_min_groups, num=None):
    fig = plt.figure()
    if num is None:
        M = m
        fig.suptitle('3D manifold Pose for Global set')
    else:
        M = m[num]
        fig.suptitle('3D manifold Pose for set ' + str(im_min_groups[num]))

    ax = plt.axes(projection='3d')
    ax.plot3D(M[0,:], M[1,:], M[2,:],'blue')
    ax.scatter3D(M[0,:], M[1,:], M[2,:],color='red')
    ax.set_xlabel('$\phi_1$')
    ax.set_ylabel('$\phi_2$')
    ax.set_zlabel('$\phi_3$')
    return


def display_eigenimages(images, u, im_indices):
    im_dim0 = np.shape(images)[1]
    im_dim1 = np.shape(images)[2]
    im_arr = []
    eig_arr = []
    
    for im_index in im_indices:
        im_arr.append(images[im_index])
        eig_arr.append(u[:,im_index].reshape(im_dim0,im_dim1))
    
    ajacent_images(im_arr)
    ajacent_images(eig_arr)
    return


def computeER(x, sv, goal=None):
    x_norm = np.linalg.norm(x)**2
    data = []
    for i in range(len(sv)):
        ER = np.sum(sv[0:i]**2) / x_norm
        if goal is not None and ER > goal:
            return i
        elif goal is None:
            data.append([i, ER])
    
    return data


def plotER(data, nums, im_min_groups, max_k = 128):
    fig = plt.figure()
    fig.suptitle('Energy recovery ratio vs. basis dimension')
    ax = plt.axes()
    legend_arr = []
    for num in nums:
        x, y = np.hsplit(np.array(data[num]), 2)
        ax.plot(x[0:max_k], y[0:max_k])
        if num < 20:
            legend_arr.append(im_min_groups[num])
        elif num == 20:
            legend_arr.append('Global Classifier')
    fig.legend(legend_arr)
    return


def print_array(array, end_char='\n'):
    print('[', end ='')
    for element in array:
        print('%7.3f' % (float(element)), end ='')
        print(',', end ='')
    print(']', end=end_char)
    return


def disp_ER_tables(k_all, k_all_list, im_min_groups):
    print('Minimum basis dimension to reach a given energy recovery for each classifier')
    print_array(k_all_list, end_char='')
    print(' - ER Ratio')
    for i in range(np.shape(k_all)[0]):
        print_array(k_all[i], end_char='')
        if i < 20:
            print(' - ' + im_min_groups[i])
        elif i == 20:
            print(' - Global')
    return


def disp_class_tables(k_all, k_all_list, data, realm, im_min_groups, desc='Average error (radians)'):
    print(desc + ' for each ' + realm + ' classifier at given Energy recovery ratios')
    print_array(k_all_list, end_char='')
    print(' - ER Ratio')
    for i in range(np.shape(k_all)[0] - 1):
        arr = [data[i,int(k)] for k in k_all[i]]
        print_array(arr, end_char='')
        print(' - ' + im_min_groups[i])
    return


def classify_object(test_image, u, m, k):
    x_dim = np.shape(u)[0]
    m_dim = np.shape(m)[1]
    
    x_new = test_image.reshape(x_dim)
    p = u[:,0:k].T @ x_new

    min_error = 0
    min_image = 0
    for i in range(m_dim):
        error = np.linalg.norm(p - m[0:k,i])
        if i == 0:
            min_error = error
            min_image = 0
        elif error < min_error:
            min_error = error
            min_image = i
    return min_image


def local_classification_avg_error(test_images, test_ans, train_ans, u, m, k_min=1, k_max=128):
    num_im = np.shape(test_images)[0]
    im_sets = np.arange(0, np.shape(test_images)[0])
    k_steps = np.arange(k_min, k_max)
    data = np.zeros((len(im_sets), len(k_steps)))
    
    for i in range(len(im_sets)):
        for j in range(len(k_steps)):
            for k in range(num_im):
                guess = classify_object(test_images[i][im_sets[k]], u[i], m[i], k_steps[j])
                data[i, j] += np.fabs((test_ans[im_sets[k]] - train_ans[guess]) / num_im)
    return data


def plot_class_error(data, nums, im_min_groups, title_addon, k_max = 128):
    fig = plt.figure()
    fig.suptitle(title_addon + ' vs. basis dimension')
    ax = plt.axes()
    ax.set_xlabel('Basis dimension')
    ax.set_ylabel(title_addon)
    legend_arr = []
    x = np.arange(0, np.shape(data)[1])
    for num in nums:
        y = data[num]
        ax.plot(x[0:k_max], y[0:k_max])
        legend_arr.append(im_min_groups[num])
    fig.legend(legend_arr)
    return


def global_classification_avg_error(test_images, test_ans, train_ans, u, m, k_min=1, k_max=128):
    num_im = np.shape(test_images)[0]
    im_sets = np.arange(0, np.shape(test_images)[0])
    k_steps = np.arange(k_min, k_max)
    data = np.zeros((len(im_sets), len(k_steps)))
    accuracy = np.zeros((len(im_sets), len(k_steps)))
    
    for i in range(len(im_sets)):
        for j in range(len(k_steps)):
            for k in range(num_im):
                guess = classify_object(test_images[i][im_sets[k]], u, m, k_steps[j])
                data[i, j] += np.fabs((test_ans[im_sets[k]] - train_ans[guess % 128]) / num_im)
                if guess // 128 != i:
                    accuracy[i, j] += 1
    return data, accuracy


def image_from_user(test_image, test_ans, train_image, train_ans, u, m, ug, mg, im_min_groups, k_all, k_all_list):
    im_sets = np.shape(test_image)[0]
    im_count = np.shape(test_image)[1]
    k_max = np.shape(m)[1]
    
    fig = plt.figure()
    fig.set_size_inches(11,8)
    fig.suptitle('Object list')
    for i in range(im_sets):
        plt.subplot(4,5,i+1)
        plt.axis('off')
        plt.title(im_min_groups[i] + ' [' + str(i) + ']')
        plt.imshow(test_image[i][0], cmap='gray')
    plt.show()
    
    set_num = int(input('Please enter a number 0-' + str(im_sets-1) + ' to select a test image set: '))
    im_num = int(input('Please enter a number 0-' + str(im_count-1) + ' to select a specific image: '))
    print('For this set, the recommended k values for a given energy recovery ration are: ')
    print_array(k_all_list)
    print_array(k_all[set_num])
    k = int(input('Please enter a number 0-' + str(k_max-1) + ' to select a subspace dimension: '))
    
    guess = classify_object(test_image[set_num][im_num], u[set_num], m[set_num], k)
    ajacent_images([test_image[set_num][im_num], train_image[set_num][guess]])
    plt.show()
    
    print('Left is the test image, right is the closest match')
    print('True pose = ' + str(test_ans[im_num]) + ', Estimated pose = ' + str(train_ans[guess]))
    error = train_ans[guess] - test_ans[im_num]
    print('Error is ' + str(error) + ' Radians, or ' + str(error*180/np.pi) + ' Degrees')
    
    yn = input('Would you also like to test the global classifier on this object? (y/n): ')
    if yn == 'y' or yn == 'Y':
        print('For the global classifier, the recommended k values for a given energy recovery ration are: ')
        print_array(k_all_list)
        print_array(k_all[20])
        k = int(input('Please enter a number 0-' + str(k_max-1) + ' to select a subspace dimension: '))

        guess = classify_object(test_image[set_num][im_num], ug, mg, k)
        guess_im = guess % np.shape(train_image)[1]
        guess_set = guess // np.shape(train_image)[1]
        ajacent_images([test_image[set_num][im_num], train_image[guess_set][guess_im]])
        plt.show()

        print('Left is the test image, right is the closest match')
        print('True set = ' + im_min_groups[set_num] + ', Estimated pose = ' + im_min_groups[guess_set])
        print('True pose = ' + str(test_ans[im_num]) + ', Estimated pose = ' + str(train_ans[guess_im]))
        error = train_ans[guess_im] - test_ans[im_num]
        print('Error is ' + str(error) + ' Radians, or ' + str(error*180/np.pi) + ' Degrees')
        return


########################## FINAL PROJECT FUNCTIONS ####################


def average_displacement(matches, kp1, kp2):
    u = [0, 0]
    std = [0, 0]
    n_matches = np.shape(matches)[0]
    for match in matches:
        q_index = match[0].queryIdx
        t_index = match[0].trainIdx
        dx = kp2[t_index].pt[0] - kp1[q_index].pt[0]
        dy = kp2[t_index].pt[1] - kp1[q_index].pt[1]
        u[0] += dx / n_matches
        u[1] += dy / n_matches
    
    for match in matches:
        q_index = match[0].queryIdx
        t_index = match[0].trainIdx
        dx = kp2[t_index].pt[0] - kp1[q_index].pt[0]
        dy = kp2[t_index].pt[1] - kp1[q_index].pt[1]
        std[0] += (dx - u[0])**2 / n_matches
        std[1] += (dy - u[1])**2 / n_matches
    
    return u, np.sqrt(std)


def overlap_images(im1, im2, dxy, opacity=0.5):
    dy, dx = int(dxy[0]), int(dxy[1])
    x_max = int(np.max([np.shape(im2)[0], dx + np.shape(im1)[0]]))
    x_min = int(np.min([0, dx]))
    y_max = int(np.max([np.shape(im2)[1], dy + np.shape(im1)[1]]))
    y_min = int(np.min([0, dy]))

    if len(np.shape(im1)) != len(np.shape(im2)):
        print('Images not both either RGB or gray')
        return
    elif len(np.shape(im1)) == 3:
        im = np.zeros((x_max-x_min, y_max-y_min, 3), dtype=np.uint8)
    elif len(np.shape(im1)) == 2:
        im = np.zeros((x_max-x_min, y_max-y_min), dtype=np.uint8)
    else:
        print('Image Error, check dimensions')
        return

    im1_x_min = np.max([0, dx])
    im1_x_max = im1_x_min + np.shape(im1)[0]
    im1_y_min = np.max([0, dy])
    im1_y_max = im1_y_min + np.shape(im1)[1]

    im2_x_min = np.max([0, -dx])
    im2_x_max = im2_x_min + np.shape(im2)[0]
    im2_y_min = np.max([0, -dy])
    im2_y_max = im2_y_min + np.shape(im2)[1]

    im[im1_x_min:im1_x_max,im1_y_min:im1_y_max] = im[im1_x_min:im1_x_max,im1_y_min:im1_y_max] + opacity * im1
    im[im2_x_min:im2_x_max,im2_y_min:im2_y_max] = im[im2_x_min:im2_x_max,im2_y_min:im2_y_max] + opacity * im2

    return im
    

def direct_blend_images(im1, im2, dxy):
    dy, dx = int(dxy[0]), int(dxy[1])
    x_max = int(np.max([np.shape(im2)[0], dx + np.shape(im1)[0]]))
    x_min = int(np.min([0, dx]))
    y_max = int(np.max([np.shape(im2)[1], dy + np.shape(im1)[1]]))
    y_min = int(np.min([0, dy]))

    if len(np.shape(im1)) != len(np.shape(im2)):
        print('Images not both either RGB or gray')
        return
    elif len(np.shape(im1)) == 3:
        im = np.zeros((x_max-x_min, y_max-y_min, 3), dtype=np.uint8)
    elif len(np.shape(im1)) == 2:
        im = np.zeros((x_max-x_min, y_max-y_min), dtype=np.uint8)
    else:
        print('Image Error, check dimensions')
        return

    im1_x_min = np.max([0, dx])
    im1_x_max = im1_x_min + np.shape(im1)[0]
    im1_y_min = np.max([0, dy])
    im1_y_max = im1_y_min + np.shape(im1)[1]

    im2_x_min = np.max([0, -dx])
    im2_x_max = im2_x_min + np.shape(im2)[0]
    im2_y_min = np.max([0, -dy])
    im2_y_max = im2_y_min + np.shape(im2)[1]
    
    im[im2_x_min:im2_x_max,im2_y_min:im2_y_max] = im2
    im[im1_x_min:im1_x_max,im1_y_min:im1_y_max] = im1
    
    return im


def direct_blend_images_H(im1, im2, dxy):
    dy, dx = int(dxy[0]), int(dxy[1])
    x_max = int(np.max([np.shape(im2)[0], dx + np.shape(im1)[0]]))
    x_min = int(np.min([0, dx]))
    y_max = int(np.max([np.shape(im2)[1], dy + np.shape(im1)[1]]))
    y_min = int(np.min([0, dy]))

    if len(np.shape(im1)) != len(np.shape(im2)):
        print('Images not both either RGB or gray')
        return
    elif len(np.shape(im1)) == 3:
        im = np.zeros((x_max-x_min, y_max-y_min, 3), dtype=np.uint8)
    elif len(np.shape(im1)) == 2:
        im = np.zeros((x_max-x_min, y_max-y_min), dtype=np.uint8)
    else:
        print('Image Error, check dimensions')
        return

    im1_x_min = np.max([0, dx])
    im1_x_max = im1_x_min + np.shape(im1)[0]
    im1_y_min = np.max([0, dy])
    im1_y_max = im1_y_min + np.shape(im1)[1]
    im1_x = np.arange(im1_x_min, im1_x_max, 1)
    im1_y = np.arange(im1_y_min, im1_y_max, 1)

    im2_x_min = np.max([0, -dx])
    im2_x_max = im2_x_min + np.shape(im2)[0]
    im2_y_min = np.max([0, -dy])
    im2_y_max = im2_y_min + np.shape(im2)[1]
    
    im[im2_x_min:im2_x_max,im2_y_min:im2_y_max] = im2
    
    for i in range(len(im1_x)):
        for j in range(len(im1_y)):
            if im1[im1_x[i]-im1_x_min,im1_y[j]-im1_y_min] != 0:
                im[im1_x[i],im1_y[j]] = im1[im1_x[i]-im1_x_min,im1_y[j]-im1_y_min]
    #im[im1_x_min:im1_x_max,im1_y_min:im1_y_max] = im1[im1 == 0]
    
    return im


def lsq_image_stitch(im1, im2, matches, kp1, kp2):
    n_matches = np.shape(matches)[0]

    A = np.zeros((0, 6))
    b = np.zeros((0, 1))
    for match in matches:
        q_index = match[0].queryIdx
        t_index = match[0].trainIdx
        A = np.vstack((A, [kp2[t_index].pt[0], kp2[t_index].pt[1], 1, 0, 0, 0]))
        A = np.vstack((A, [0, 0, 0, kp2[t_index].pt[0], kp2[t_index].pt[1], 1]))
        b = np.vstack((b, [[kp1[q_index].pt[0]], [kp1[q_index].pt[1]]]))
    return np.array(LA.inv(A.T @ A) @ (A.T @ b)).reshape((2, 3))


def warp_perspective_blend(im, im2, H, max_size=(5000,5000)):
    x_add = 0
    y_add = 0
    
    x_in_max = np.shape(im)[0]
    y_in_max = np.shape(im)[1]
    out_00 = ([0, 0, 1]) @ H.T
    out_m0 = ([x_in_max, 0, 1]) @ H.T
    out_mm = ([x_in_max, y_in_max, 1]) @ H.T
    out_0m = ([0, y_in_max, 1]) @ H.T
    out = np.array([out_00, out_m0, out_mm, out_0m])
    x_max = np.max(out[:,0])
    x_min = np.min(out[:,0])
    y_max = np.max(out[:,1])
    y_min = np.min(out[:,1])
    
    if x_min < 0:
        x_add = -x_min
        x_max -= x_min
        x_min = 0
    elif y_min < 0:
        y_add = -y_min
        y_max -= y_min
        y_min = 0
    x_max = np.max([x_max, max_size[0]])
    y_max = np.max([y_max, max_size[0]])
    
    H[0][2] += x_add
    H[1][2] += y_add
    im_warp = cv2.warpPerspective(im, h, (5000, 5000))
    
    direct_blend_images(im_warp, im2, [x_add, y_add])
    return


def sift_good_matches(img1, img2, threshold=0.5):

    # Code essentially copied from: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    print('Got ' + str(np.shape(matches)[0]) + ' matches')

    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    print('Got ' + str(np.shape(good)[0]) + ' good matches')
    return good, kp1, kp2


def orb_good_matches(img1, img2, threshold=0.5):

    # Code essentially copied from: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    print('Got ' + str(np.shape(matches)[0]) + ' matches')

    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    print('Got ' + str(np.shape(good)[0]) + ' good matches')
    return good, kp1, kp2


def panoramic(images, method='direct', detector='sift', threshold=0.5):
    num_im = np.shape(images)[0]
    image = np.copy(images[0])
    for i in range(1, num_im):

        if detector == 'sift':
            good, kp1, kp2 = sift_good_matches(image, images[i], threshold)

        elif detector == 'orb':
            good, kp1, kp2 = orb_good_matches(image, images[i], threshold)

        if method == 'direct':
            if np.shape(good)[0] < 4:
                print('Too few keypoints, try a higher threshold')
                return image
            u, std = average_displacement(good, kp1, kp2)
            print('mean: ' + str(u) + ' STD: ' + str(std))
            image = direct_blend_images(image, images[i], u)

        elif method == 'affine':
            if np.shape(good)[0] < 4:
                print('Too few keypoints, try a higher threshold')
                return image
            u, std = average_displacement(good, kp1, kp2)
            print('mean: ' + str(u) + ' STD: ' + str(std))
            R = lsq_image_stitch(image, images[i], good, kp1, kp2)
            im_affine = cv2.warpAffine(images[i], R, (2000, 1500))
            image = direct_blend_images_H(image, im_affine, [0, 0])

        elif method == 'homography':
            if np.shape(good)[0] < 4:
                print('Too few keypoints, try a higher threshold')
                return image
            u, std = average_displacement(good, kp1, kp2)
            print('mean: ' + str(u) + ' STD: ' + str(std))
            src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)

            h, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            im_hom = cv2.warpPerspective(images[i], h, (2000, 1500))
            image = direct_blend_images_H(image, im_hom, [0, 0])

    return image
