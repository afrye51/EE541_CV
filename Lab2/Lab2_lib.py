import skimage
from skimage import io
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from scipy import signal
from skimage.transform import resize
import scipy.ndimage.filters as fi


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


def detect_corners(image, alpha=0.04, show=False):
    k = gaussian_2d_order(5, 1, (0,1))
    k2 = gaussian_2d_order(5, 2, 0)
    i_x = signal.convolve2d(image, k)
    i_y = signal.convolve2d(image, k.T)
    i_xy = i_x*i_y
    i_xx = i_x**2
    i_yy = i_y**2
    i_xy2 = signal.convolve2d(i_xy, k2)
    i_xx2 = signal.convolve2d(i_xx, k2)
    i_yy2 = signal.convolve2d(i_yy, k2)
    result = i_xx2*i_yy2 - i_xy2**2 - alpha * (i_xx2 + i_yy2)**2
    if show:
        ajacent_images([i_x, i_y])
        ajacent_images([i_xx, i_yy, i_xy])
        ajacent_images(result)
    return result