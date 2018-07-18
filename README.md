[img1]: latest_run/um_000000.png
[img2]: latest_run/um_000010.png
[img3]: latest_run/um_000020.png
[img4]: latest_run/um_000030.png
[img5]: latest_run/um_000040.png
[img6]: latest_run/um_000050.png
[img7]: latest_run/um_000060.png
[img8]: latest_run/um_000070.png
[img9]: latest_run/um_000080.png
[img10]: latest_run/um_000090.png


# Semantic Segmentation

### Overview

This project is part of the Udacity Self-Driving Car Engineer Nanodegree. This project is the Advanced Deep Learning elective. The goal of this project is to build and train an FCN-8 semantic segmentation neural network to detect driveable areas of a scene from images (roadways).

A semantic segmentation network is an encoder / decoder network connected by a one by one convolution. The network can classify target classes in an image on a pixel by pixel basis, unlike a CNN that loses location information during classification.

### Approach

In order to build out our FCN-8 we use an existing VGG-16 network as the encoder, attached a 1x1 convolution and then build out the encoder. While building out the encoder we also add in skip layers which are direct connections from certain layers in the encoder to certain layers in the decoder. Skip layers allow the network to better deal with different resolutions of objects within the input image.

The most difficult part of building the network for me was to figure out how to connect the skip layers to the decoder layer, as the skip layer and decoder layers had different numbers or kernels. I dealt with this by adding a 1x1 convolution between each of them.

### Training

I used the Adam Optimizer to train the network. I also added a value of 0.5 for the dropout to the VGG network. L2 regularization was added to each layer of the decoder sa suggested by Udacity. I first set the initial weights of the decoder layers by using a Xavier initializer but I found I had much better luck with using a normal distribution. I tried many values for batch and epoch and learning rate. I managed to get the batch size up to 4 by training on an AWS GPU server with a high end graphics card. The learning rate had to be quite small I found to get the loss to continue decreasing. Settling on a value of 0.00003. I ended up setting the number of epochs to around 10. I did not try training beyond that number, but I found that less than this left a very grainy image (poor classification).

### Inference

After training several input images were sent through the network. The network outputs pixels that it beleives are driveable (roadway). These were then plotted back on the input image as green pixels (see below a selection of images from the network):

![img1]
![img2]
![img3]
![img4]
![img5]
![img6]
![img7]
![img8]
![img9]
![img10]

### Improvements

I could further improve the project by adding IoU results to gauge just how accurate the inferences are. I would also like to play further with the learning parameters to see if I can further improve the network's classification accuracy.

---
### Herefollows The Original Udacity Project Brief and Instructions...

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
