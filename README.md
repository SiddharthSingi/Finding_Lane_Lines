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
1. Convert to  grayscale.
1. Use Canny Detection on the grayscale image.
1. Apply a Region of interest on the image.
1. Find Lines using Hough transform
