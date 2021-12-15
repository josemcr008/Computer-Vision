



# SESION 1

## Colors descritors
Para diferenciar una imagen rotada(bandera que sea la misma rotada), dividimos la imagen en regiones y se calcula el histograma de cada region. A estos se les llama **local descriptors**.


## Texture descriptors
 * Vector con cosas: media, varianza, entropia...
 * We can use the gradients(another vector). We use the derivatives. For each pixel i have the magnitude and the orientation. We get the histogram(again a vector) in which we can see the grades of the lines.
 

## Shape descriptors
* Una forma con una fórmula chunga




# Assignment 1
* We are going to use a method based on image difference. Basically is calculate the difference between the frame t and t+1. This method is a bit noisy. To remove the noise we use opening+closing. 
* We can use another method. The diffence between the image and the background. To make this more robust, we estimate a gaussian model (mean, standar derivation). 


# SESION 2
 
 * Gradient: describe a direccional change of the instenstiy of an image -> Sobel filter.  We can obtein the **shape** of the differents objects that we have on the image.  `orientation = atan(gy/gx)`

 * Laplacian-of-Gaussian. We obtein the edges of the object

0  1  0
1 -4  1
0  1   0

Another way to compute this is using the difference of gaussian

    LoG = G2 - G1 // G2 and G1 must have differents values of sigma

## Canny edge detectors
* Improve the edge cuality in order to find. We obtain Em(magnitude) and Ed(direction)
* Non-maxima suppresion: reduce the width of the edges to one pixel.
	* Discretize the gradient directions in 4 main directions.
	* Create a new image Eg' whose value are 1,2,3,4 according to the nearest angle.
	* Scan Em(matrix that has the magnitude of the image) and Eg'  to create a new image Em' as follows:
		* Em'(x,y)= 0 if Em(x,y) is smaller than at least one of the neighbours in the direction normal to the gradient
		* Em'(x,y=)= Em(x,y) otherwise.
* Hysteresis: we use 2 threshold. tl < tu.
	* pixel > tu -> edge
	* pixel < tl -> no edge
	* tu < pixel < tl -> edge if the pixel is conected with a valid pixel (checking the gradient direction)

How to select the thresholds? select threshold as percentiles of the grey-level distribution

# Sesion 3

## Optical flow
A pattern of apparent motion of the object. We use to obtain objects that are moving in two images. OF is a vector. Three assumptions:
 * Brightness does not change from frame to frame.
 * Small movements.
 * Neighboring points belong to sme surface and have similar motion.

`I(x(t),t) = I(x(t+dt),t+dt)`

I = intensity
x = spacial position
t = time  
dt = increment of time

two points : (2,10) , (4,12)
The optical flow will be: (2,10) - (4,12) = (2,2) = (u,v)


# Sesion 4

## Segmentation based on colors -> We need only one image

* Typical green background.

## Segmentation based on optical flow 
	
* Optical flow is used to represent movement. 
* Is used to blur the background of an image(the static pixels) and we can use it to read for example the movement of a person's hand.
* Basic method:
	* 1- Compute the optical flow between sequential frames: t-1 with 1.
	* 2- For each pixel, compute maginute of optical flow.
	* 3- NOrmalize OF magnitude in [0,1]
	* 4- If Mag(y,x) > T, then "(y,x) is foreground"


# Sesion 5

## Machine Learning

Is the subfield of computer science that, according to Arthur Samuel in 1959, gives "*computers the ability to learn without being explicity programmed*"
Clasisificaction is a general procces

### Classfication: Ingredients
* Dataset: contains samples and labels.
* Splits: 
	* Training: to learn parameters of the model (clasiffier)
	* Validation: to tune hyperparameters, eg K in kNN.
	* Test: to predict performance at deployment time.

### Classification: Metrics
* Accuracy: how often the classifier makes the correct prediction.

     Acc=( #correct / #total ) * 100

* Confusion matrix: shows details about correct and incorrect classifications for each category

    | ## | P#1  | P#2 | P#3 |	
    |T#1 | 14    | 1     | 0      |
    |T#2  | 31   | 15   | 1      |			
    |T#3  | 1      | 2     | 12    |
    
    Normalice
    
    | ## | P#1  | P#2 | P#3 |
    |T#1 | 93,3    | 6,7     | 0      |
    |T#2  | 66      | 31,9   | 2,1   |
    |T#3  | 6,7     | 13,3    | 80   |

### A very simple cassifier

We need a training set:
people(category) vs background(category)

PEOPLE 
S1: | _ |
s2:  | _ |

T1: | _ | 

Idea: compute de distance between target and training samples. E.g Chi-square distance.
Distance between:
T1 ---> s1. s2, b1, b2

    |						 x = backgroung
    |                            *                   * = people
    |	  x	x     d   *	*  *
    |	x                           *
    |	   x
    |__________________________________________________

Using knn
with k= 3



### Typeos of classifiers
* Supervised(labels to every sample) 
* Semi-supervised(labels to samples but not to everyone) 
* Unsupervised(no labels)

### Machine Learning in OpenCV
Parent class: cv::ml::StatModel

Selected mehods:
* isTrained() 
* train(trainSamples, ROW_SAMPLE, labels)
* predict(testSamples, predictions)
* getVarCount() // To provide information

### SVM : Support Vector Machines

Autocamitally selected samples define a hyper-plane to separate postive from negative samples. -> binary classifier( only can two classes)
Margin -> parameter. The separation between the two classes.

    |						 
    |               |   *                   
    |	 x x   | *  *
    |	x     |            *
    |	   X |
    |_____________________________


    |						 
    |            *                         
    |	*  x  x  x
    |    *	 x  x x     *
    |	   * *   
    |_____________________________

For this example, we make a higher dimensional space. 
Several types of kernel:
* Linear
* Polynomial
* Radial basis -> exponential function.
* Sigmoid -> _/-

### Pedestrian detector
* Features : Histogram of Gradients.
* Given a dataset, split into: 
	* Training; learn SVM parameters.
	* Validation: tune hyper-parameters

Training + validation -> Test
Validation to chose the hyperparatemeters and train the classifier with those parameters. 

## Multiclass 
From binary classfiers to multiclass:
* One vs all
	* Take the arg max
		* f1(x): C1 vs {C2,C3}
		* f1(x): C2 vs {C1,C3}
		* f1(x): C3 vs {C2,C1}
* One vs one
	* Take the arg max
		* f1(x): C1 vs C2
		* f1(x): C1 vs C3
		* f1(x): C2 vs C3

		# Sesion 6

### Boosting

    |		*				 
    |      _ _ _ _ _                          
    |   *  |x  x  x |
    |    * | x  x x |    *
    |      |        |
    |_____________________________		(each line is a different classifier)

Using different clasffiers to make one strong classifier. Define to binary problems. 

 Example: Viola & Jones face detector

### Decision Trees

    |						 
    |                |    *  *                       
    |	  x  x  x|
    | t2   	-------- |- - - - ----             
    |	   * *   |	x x x
    |_____________________________
			t1

x>T1?  -> no / y>T2 -> no -> class B / -> yes -> class A

#### Random forest
Many small decision trees on randomly sleected dimensions. Final output -> majority voting

| tree |  | | 
|--|--|--|--|
| 1 | x |*|x|
| 2| x | *|
| 3| x|x
|result|x|*


## Trends on Feature Learning

### Deep Learning
Neural Networks with a lot of layers thar are conected with a lot of parameters.

    0-->0-->0-->0-->0-->0
    0-->0-->0-->0-->0-->0
    0-->0-->0-->0-->0-->0
    0-->0-->0-->0-->0-->0
    0-->0-->0-->0-->0-->0	Every layer should be connected with every of the following col.
Layers of the inside are called hidden layers.

### CNN
* Deep, feed-forward network with convolutional layers. Filters are learnt automatically form images/videos.
Max pool: compute the maximun of the window. We use this to reduce the size of the image.

LeNet (diapositiva: A simple CNN)

### Aplications
* Content generation
* Automatic image generation
* Video synthesis
* Lip Reading Sentences
* Image captioning
* Medical image analysis
* Autonomus cars


# Sesion 7

### Categoritzation
Choos e a category from a set: "The **wat**, but **NOT** the **where**"

### Detection/Localisation
Define the image region of the objects of interest: "The **what** and the **where**"

### Semanctic segmenentation
Assign one label to each pixel of the image

## Categorization

#### Why is difficult?
* Intra-class variability. We can have a lot of differents chairs (for example).
* Inter-class variability: Different classes that shares some caracteristics. (a horse and a zebra)
* Camera viewpoint
* Illumination
* Oclussion: some parts of the objects are not visible.
* Clutter: many objects in the foreground.
* Deformation: objects changing their position (a human can move their arms)

## Convolutional Neural Networks
 Deep. feed-forward network with convolutional layers. Filsters are learn automatically from images/videos.

## Layers

### Convolution
2D Convolutional layer:
* Filter size : e.g 3x3
* Number of filters
* Stride: how many pixels you move to one side.
* Padding: valid, same
* Activation: ReLU. once we have the convolution done., the output is called feature map or activation map as well. Activation function:
	* ReLU: if we have a negative value, we change to 0, the positive stays equal.  
	* sigmoid: the min value is 0 and max is 1.

### Dense
A.k.a Fully- connected layer
Parameters:
* Number of units
* Activation

### The output
Dense layer with softmax activation. As many units as categories.

    Output layer			 Softmax activation function						Probabilites
    [1.3]																			[0.02]
    [5.1]			-->												--> 			[0.90]
    [2.2]																			[0.05]

# Sesion 8

## Sequential model
Combine the pieces. Tips:
* Several Conv2D+MaxPool2D bocks, incremental numbers of filters.
* Some FC layers decreasing their size.
* Final FC layers with softmax
Labels: one-hot vector encoding. Relation between output layer and this encoding
One-hot vector -> means only one postion of the vector is 1 and the other are 0.
The output we will recieve is not perfect, for example we obtain (0,1,0,0,0,1). We want to optimize that.

Model has to be compiled. To be set:
* loss: image -> output  -> target. We want to minimize the loss function as much as posible. Loss is how good is our output comparing to the target.
* optimizer: 
* metrics: accuracy 


# Sesion 9

## Training

### Data batch
A lot of images with a specific width and height. Batch size is an hyperparameter

Epoch: training period where the model sees (almost) all training samples. Hyperparameter.
Usually we shuffle data between epochs.

We can make a graphic using the accuracy and the loss function. If the accuracy increase, the loss should be decreasing. 

### Hyprparameters
* Data: batchs size. Depending on the GPU and the images.
* Optimizer: type and learning rate (we use the gradient with respect to the loss function. Reach the minimun value of the loss function) We decreease the learning rating when the validation loss decrease.
* Number of epochs: two posibilties:
	* Stop training too early or too late.
	* Traing during a lot of epochs and when the loss function stucks, stop training,

Example:

| bs | ot |lr |ep|
|--|--|--|--|
| 16 | SGD |0.01|30|
| 32 | SGD |0.01|30|
| 16 | Adam |0.01|30|
| . | . |.|.|
| . | . |.|.|
| . | . |.|.|

We use the validation accuracy or loss function.


# Sesion 10

## Overfitting

The network fits its parameters extremely well on the training data.
* Very high Acc on training
* Acc starts decreasing for the validation set --> loss increases
Poor genereralization on test data. Is a situation that we must avoid.

In order to avoid this, we can:

* More data:
	* Gather more data.
	* Data augmentation techniques (make different pertubation to an image to get more images)

* Regularization
	* New hyperparameter: drop ratio.
	* Regularization parameter of layers.

* Transfer learning
	* Reuse layers and weights from other pretrained network.
		* Freeze some layers and train the others

Input --> CNNa --> Class(a) --> Output
Input --> CNNa --> Class(b) --> Output


# Sesion 11

## Keypoint detection

Keypoint is limite region, not a single point. This shpuld be easy to recognize and largely unique. Is defined by the x,y,s --> position and the scale.
Important: the methpd we use must be invariant to translation and rotation.

### SIFT

Scale-Invariant Feature Transform
Steps:

* Pyramid of scales. (same image with different resolution)
* Convolution with Gaussians --> variance.  
* Computing the difference of Gaussians is a Laplacian of Gaussian filter(detect edges)
* Selection of maxima and minma in Difference of Gaussian. Maximum value of the window/neighbours in the same image, the scale above and the scale below. Same with the minimun value.
* Reject points of the edges and low contrast.
* keypoint scale --> square root of the smallest variance used in this DoG.
* Keypoint orientation --> peak in histogram of orientations of neighbour points. (1.5 x scale)


## Descriptors

Mathematical representation image regions associated to keypoits. Two types: real or binary values.

Computing steps:

* Split region into 16 blocks.
* For each block, compute the histogram of gradient orientations (8 bins). Contribution of each pixel proportional to magnitude --> 128 dims.
* Rotation wrt keypoint orientation.
* Vector normalization.

## Marching
Finding pairs of keypoints in different images.
Distance between descriptors.
* Euclidean
* Hamming (amount of bits that are different. It used if we use the binary descriptor)

Classic strategy: assigning to each descriptor its Neirest Neighbour, Alternative FLANN (Fast Library for Approximate Nearest Neighbours)

Pair filtering --> distance above a threshold. Reject points that are not so good.

## Aplications

### Object localization
* Define the image region of a known object.
* Idea: use the homography(transformation between two planes) to compute the position of the corners of the objects in the scene.

### Stitching
 Linking images of same scene to generate a single image (panorama)


