# Omar El Nahhas
# ITS8030 - Computer Vision
# 21/09/2021

"""
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

Please note that NumPy arrays and PyTorch tensors share memory represantation, so when converting a
torch.Tensor type to numpy.ndarray, the underlying memory representation is not changed.

There is currently no existing way to support both at the same time. There is an open issue on
PyTorch project on the matter: https://github.com/pytorch/pytorch/issues/22402

There is also a deeper problem in Python with types. The type system is patchy and generics
has not been solved properly. The efforts to support some kind of generics for Numpy are
reflected here: https://github.com/numpy/numpy/issues/7370 and here: https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY
but there is currently no working solution. For Dicts and Lists there is support for generics in the 
typing module, but not for NumPy arrays.
"""
import cv2
import numpy as np
from util import gkern, gkern1D
import math
import random

"""
Task 1: Convolution

Implement the function 

convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray

to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:

    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively

(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)

    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is True, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""
def convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int,
                kernel_height : int, add : bool, in_place : bool = False) -> np.ndarray :
    if not in_place:
        image = image.copy()

    if image.ndim == 3:
        (iH, iW, colour_channels) = image.shape
        output = np.zeros((iH, iW, colour_channels), dtype="float32")
    else:
        (iH, iW) = image.shape
        colour_channels = 0
        output = np.zeros((iH, iW), dtype="float32")    

    # Check parameters
    if(kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0): #even number
        print("Error! Kernel dimensions are even, but require to be odd.\n")
        return 
    
    kH = kernel_height
    kW = kernel_width

    # calculate padding
    pad_tb = (kW - 1) // 2  
    pad_lr = (kH - 1) // 2 

    # Convolution
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    image_pad = cv2.copyMakeBorder(image, pad_tb, pad_tb, pad_lr, pad_lr,
        cv2.BORDER_CONSTANT, value=0)


    for i in range(iH):
        center_x = pad_tb + i
        indices_x = [center_x + l for l in range(-pad_tb, pad_tb  + 1)]
        for j in range(iW):
            center_y = pad_lr + j
            indices_y = [center_y + l for l in range(-pad_lr, pad_lr + 1)]
            submatrix = image_pad[indices_x, :][:, indices_y]

            if colour_channels == 3: #RGB
                for channel in range(colour_channels):
                    output[i][j][channel] = np.sum(np.multiply(submatrix[:, :, channel], kernel)) + add * 128
            else:
                output[i][j] = np.sum(np.multiply(submatrix, kernel)) + add * 128

    return output


"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray 

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""
def gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2*radius + 1

    kernel = gkern(kernlen=kernel_size, nsig=sigma)
    output = convolution(image, kernel, kernel_size, kernel_size, add = False, in_place=False)

    return output



"""
Task 3: Separable Gaussian blur

Implement the function

separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given normalize_kernel() function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
"""
def separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :  
    

    if not in_place:
        image = image.copy()

    radius = int(math.ceil(3 * sigma))
    kernel_size = 2*radius + 1
    #kernel_size = 3 #just for checking, remove after

    kernel_hori = (gkern1D(kernlen=kernel_size, nsig=sigma)).T
    kernel_vert = gkern1D(kernlen=kernel_size, nsig=sigma)


    print(kernel_hori.shape)
    print(kernel_vert.shape)

    #horizontally
    horiz_conv = convolution(image, kernel_hori, kernel_hori.shape[0], kernel_hori.shape[1], add = False, in_place=False)

    #vertically
    vert_horiz_conv = convolution(horiz_conv, kernel_vert, kernel_vert.shape[0], kernel_vert.shape[1], add = False, in_place=False)

    return vert_horiz_conv


"""
Task 4: Image derivatives

Implement the functions

first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray
first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray and
second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray

to find the first and second derivatives of an image and then Gaussian blur the derivative
image by calling the gaussian_blur_image function. "sigma" is the standard deviation of the
Gaussian used for blurring. To compute the first derivatives, first compute the x-derivative
of the image (using the horizontal 1*3 kernel: [-1, 0, 1]) followed by Gaussian blurring the
resultant image. Then compute the y-derivative of the original image (using the vertical 3*1
kernel: [-1, 0, 1]) followed by Gaussian blurring the resultant image.
The second derivative should be computed by convolving the original image with the
2-D Laplacian of Gaussian (LoG) kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] and then applying
Gaussian Blur. Note that the kernel values sum to 0 in these cases, so you don't need to
normalize the kernels. Remember to add 128 to the final pixel values in all 3 cases, so you
can see the negative values. Note that the resultant images of the two first derivatives
will be shifted a bit because of the uneven size of the kernels.

To do: Compute the x-derivative, the y-derivative and the second derivative of the image
"cactus.jpg" with a sigma of 1.0 and save the final images as "task4a.png", "task4b.png"
and "task4c.png" respectively.
"""
def first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([[-1, 0, 1]])
    (kH, kW) = kernel.shape
    dx = convolution(image, kernel, kH, kW, True)
    return gaussian_blur_image(dx, sigma)


def first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([[-1], [0], [1]])
    (kH, kW) = kernel.shape
    dy = convolution(image, kernel, kH, kW, True)
    return gaussian_blur_image(dy, sigma)


def second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    (kH, kW) = kernel.shape
    ddx = convolution(image, kernel, kH, kW, True)
    return gaussian_blur_image(ddx, sigma)



"""
Task 5: Image sharpening

Implement the function
sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray
to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
"""
def sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray :
    second_derivative = second_deriv_image(image, sigma, in_place=True) - 128
    image = image - second_derivative*alpha
    return image


"""
Task 6: Edge Detection

Implement 
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""
def sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray :
    if not in_place:
        image = image.copy()
    
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # print(np.mean(image))

    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    x_dir = convolution(image, k_x, k_x.shape[0], k_x.shape[1], False)/8.


    k_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    y_dir = convolution(image, k_y, k_y.shape[0], k_y.shape[1], False)/8.

    #print(y_dir)
    magnitude = np.hypot(x_dir, y_dir)
    direction = np.arctan(np.divide(x_dir, y_dir))
    return magnitude, direction


"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""
def bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray :
    iH, iW, *_ = image.shape
    x1 = math.floor(x)
    y1 = math.floor(y)

    x2 = math.ceil(x)
    y2 = math.ceil(y)
    rgb = 0.0

    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return rgb

    if x1 >= iH or x2 >= iH or y1 >= iW or y2 >= iW:
        return rgb

    #https://en.wikipedia.org/wiki/Bilinear_interpolation
    q11 = image[x1][y1]
    q12 = image[x1][y2]
    q21 = image[x2][y1]
    q22 = image[x2][y2]

    a = (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
            )
    b = ((x2 - x1) * (y2 - y1) + 0.0) #make a float

    if b != 0:
        rgb = a / b
    return rgb



def rotate_image (image : np.ndarray, rotation_angle : float, in_place : bool = False) -> np.ndarray :
    "from lab"
    radians = math.radians(rotation_angle)
    image_copy = np.zeros_like(image)
    image_height, image_width, *_ = image.shape
    for r in range(image_height):
        for c in range(image_width):
            x0 = c - image_width/2.0
            y0 = r - image_height/2.0
            x1 = x0 * math.cos(radians) - y0 * math.sin(radians)
            y1 = x0 * math.sin(radians) + y0 * math.cos(radians)
            x1 += image_width/2.0
            y1 += image_height/2.0
            rgb = bilinear_interpolation(image, y1, x1)
            image_copy[r][c] = rgb
    return image_copy
 
"""
Task 8: Finding edge peaks

Implement the function
find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray
to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinear_interpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
"""
def find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray :
    magnitude, direction = sobel_image(image)
    iH, iW, *_ = image.shape

    dst_image = np.zeros_like(image)

 
    for c in range(0, iW):

        for r in range(0, iH):
            angle = direction[c][r]
            if math.isnan(angle):
                angle = 0

            #from task 8 description
            e1x = c + 1 * np.cos(angle)
            e1y = r + 1 * np.sin(angle)
            e2x = c - 1 * np.cos(angle)
            e2y = r - 1 * np.sin(angle)

            e = magnitude[c][r]
            e1 = bilinear_interpolation(magnitude, e1x, e1y)
            e2 = bilinear_interpolation(magnitude, e2x, e2y)

            if e > thres and e > e1 and e > e2:
                dst_image[c][r] = 255.
    
    return dst_image



"""
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function

random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray

to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "num_clusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.randint(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.

To do: Perform random seeds clustering on "flowers.png" with num_clusters = 4 and save as "task9a.png".
"""
# adapted from https://medium.com/analytics-vidhya/image-segmentation-using-k-means-clustering-from-scratch-1545c896e38e
np.random.seed(1337)

class random_seed_image():
    def __init__(self, num_clusters=4, max_iters=100, epsilon=30):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.epsilon = epsilon
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.num_clusters)]
        # the centers (mean feature vector) for each cluster
        self.cntrs = []
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize random seed
        random_sample_idxs = np.random.randint(255, size=(self.num_clusters, 3))
        self.cntrs = [idx for idx in random_sample_idxs]
        print(self.cntrs)
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest cntrs (create clusters)
            self.clusters = self._create_clusters(self.cntrs)

            # Calculate new cntrs from the clusters
            cntrs_old = self.cntrs
            self.cntrs = self._get_cntrs(self.clusters)
            
            # checnum_clusters if clusters have changed
            if self._is_converged(cntrs_old, self.cntrs):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_lab(self.clusters)
    def _get_cluster_lab(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        lab = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                lab[sample_index] = cluster_idx
        return lab
    def _create_clusters(self, cntrs):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, sample in enumerate(self.X):
            cntr_idx = self._closest_cntr(sample, cntrs)
            clusters[cntr_idx].append(idx)
        return clusters
    def _closest_cntr(self, sample, cntrs):
        # dist of the current sample to each centroid
        ds = [L1(sample, point) for point in cntrs]
        closest_index = np.argmin(ds)
        return closest_index
    def _get_cntrs(self, clusters):
        # assign mean value of clusters to centroids
        cntrs = np.zeros((self.num_clusters, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            cntrs[cluster_idx] = cluster_mean
        return cntrs
    def _is_converged(self, cntrs_old, cntrs):
        # dist between each old and new centroids
        ds = [L1(cntrs_old[i], cntrs[i]) for i in range(self.num_clusters)]
        return sum(ds) < self.epsilon*self.num_clusters

    def cent(self):
        return self.cntrs

def L1(x1, x2):
    return np.sqrt(np.sum(np.abs((x1 - x2))))
"""
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function
pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False)
to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "num_clusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
"""
# adapted from https://medium.com/analytics-vidhya/image-segmentation-using-k-means-clustering-from-scratch-1545c896e38e
class pixel_seed_image():
    def __init__(self, num_clusters=5, max_iters=100, epsilon=30):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.epsilon = epsilon
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.num_clusters)]
        # the centers (mean feature vector) for each cluster
        self.cntrs = []
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize random seed
        random_sample_idxs = np.random.choice(self.n_samples, self.num_clusters, replace=False)
        self.cntrs = [self.X[idx] for idx in random_sample_idxs]
        print(self.cntrs)
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest cntrs (create clusters)
            self.clusters = self._create_clusters(self.cntrs)

            # Calculate new cntrs from the clusters
            cntrs_old = self.cntrs
            self.cntrs = self._get_cntrs(self.clusters)
            
            # checnum_clusters if clusters have changed
            if self._is_converged(cntrs_old, self.cntrs):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_lab(self.clusters)
    def _get_cluster_lab(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        lab = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                lab[sample_index] = cluster_idx
        return lab
    def _create_clusters(self, cntrs):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, sample in enumerate(self.X):
            cntr_idx = self._closest_cntr(sample, cntrs)
            clusters[cntr_idx].append(idx)
        return clusters
    def _closest_cntr(self, sample, cntrs):
        # dist of the current sample to each centroid
        ds = [L1(sample, point) for point in cntrs]
        closest_index = np.argmin(ds)
        return closest_index
    def _get_cntrs(self, clusters):
        # assign mean value of clusters to centroids
        cntrs = np.zeros((self.num_clusters, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            cntrs[cluster_idx] = cluster_mean
        return cntrs
    def _is_converged(self, cntrs_old, cntrs):
        # dist between each old and new centroids
        ds = [L1(cntrs_old[i], cntrs[i]) for i in range(self.num_clusters)]
        return sum(ds) < self.epsilon*self.num_clusters

    def cent(self):
        return self.cntrs