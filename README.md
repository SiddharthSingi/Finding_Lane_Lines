## Finding Lane Lines
![output_solidwhitecurve](https://cloud.githubusercontent.com/assets/26694585/26761841/7dbb00e0-4954-11e7-8fb4-cf0b63a7bb5a.jpg)


When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.

**Contents**
* Readme.md
* P1.ipynb
* _test_output folder_
* white.mp4

# Pipeline Description
I have used a very simple pipleline to identify the lanes in my video. The modules used in the pipeline are mainly  matplotlib, numpy and cv2. These are the steps taken in the pipeline:
1. Convert to  grayscale:
1. Apply Gaussian Blur.
1. Use Canny Detection on the grayscale image.
1. Apply a Region of interest on the image.
1. Find Lines using Hough transform
1. Display the final image.

# The main part of the code used for this is:
```
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    """Applies the Canny transform"""    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
    

    """Applies a Gaussian Noise kernel"""
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    
    Applies an image mask.
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)       
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,thick):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines,thickness=thick)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
def process_image(image):
    img_copy = copy.copy(image)
    gray = grayscale(image)
    #plt.imshow(gray)
    smooth_edges = gaussian_blur(gray,9)
    edges = canny(smooth_edges,60,180)
    vrtcs = np.array([[(100,550),(380,300),(500,300),(900,550)]])
    mapped_img = region_of_interest(edges,vrtcs)
    lineimg = hough_lines(mapped_img,1,1*np.pi/180,35,5,20,5)
    result=weighted_img(lineimg,image)
    return result
    
```



