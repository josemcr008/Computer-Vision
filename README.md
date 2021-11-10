# Computer Vision

* ## Image Basic
* ## 3D Image
* ## Convolutional Neural Network

## Assignments

 1. Create a program that loads an RGB image and shows a basic GUI to select an image region (a rectangle) with the mouse. When two points in the image are selected, the program will show another window where the inner rectangle region is in color, while the rest is in greyscale (see Figure below). The user should be able to repeat the operation as many times as desired without restarting the application.
 2. Make a program that reads an image and equalizes it. The program can not use the cv::equalizeHist function. You have to do everything yourself.
 3. Implement a NxN box filter for an image (where N is odd). Make a program that reads an image in grey scale, apply a box filter and save the result. 
 4. Write a prgram that reads an image from file, converts it to gray scale and computes the magnitude of the derivative using the Sobel filter. You have to use your own implementation of applyKernel, based on the one you did for the previous assigment. In order to avoid overflow problems, you will convert the input grey scale image into a floating point image and will do the convolution. In this case, both the image and the kernel will be floating point pixels. The resulting magnitude image, will need to be transformed into a grey scale image. You will have to scale the magnitude into a [0,255] uchar image.
 5. Create a program that reads and image and a value for the High Boost filter and saves the corresponding image with the [HighBoost](https://moodle.uco.es/m2122/mod/resource/view.php?id=158581 "highboost") Filter applied
 6. Create a program that compares the result of the median filter and the box filter for the input image given.
 7. Create a program that reads an input image, threshold the image, and then applies the morphological operations: erode, dilate, open, close, saving the result into a file. The program must have the following command-line arguments: ./prog in_image out_image -thres <val> -op <(erode|dilate|open|close)>
