# IAPR-project
The code files for the Image Analysis and Pattern Recognition course project at EPFL 
With collaboration of : Zeno Messi and Deyanira Graciela Cisneros Lazaro 

### Main Task 
Analyze the trajectory of a robot in a video sequence, and retrieve the formula and its answer and output it in a new video step by step.

![alt text](https://github.com/aymenbahroun/IAPR-project/blob/main/iapr_output_example.png?raw=true)

### Segmentation of digits and operators 

* Retrieve a denoised image without the robot
* Normalization
* Thresholding of the image¨
* Dilation to get rid of central line 
* Labelling of digits and operators 

### Segmentation of the robot

* Object labelling algorithm
* Clear connected objects to image border 
* The arrow on the robot is the biggest object
* retrieve the box surrounding the arrow object to segment the robot

### Intersection of the robot and the numbers/operators 

* At each frame :
  * Calculate the intersection of the different objects with the robot
  * Detect the intersection with a value equal to the object area
  * Go to a digit or operator classification task (depending on the evolution of the equation)

### Classification of operators 

* Equal and divide sign have more than a region (2 and 3 respect.)
* For the '+', '-', and 'x' signs :
  * They have different symmetries 
  * 2-Fold, 4-Fold, and 6-Fold symmetries 
  * Rotate the operators 360 times 
  * Fourier transform : recognize number of axes of symmetries 
  
### Classification of digits 

* We designed a 13-layer CNN and trained it on mnist dataset 
* 99.6% accuracy on test-set 
* For the rotated digits :
  * Align data (principal axis of inertia vertical)
  * Take the 180 degrees equivalent 
  * Avoid training for each rotation 
  * 95% accuracy on test set 
