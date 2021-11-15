



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