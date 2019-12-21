# Object-Detection-Recognition-using-ResNet
A Simple Object Recognition using ResNet152 trained on imagenet, tiny_imagnet modified to be trained on custom sized image dataset

There are 3 models to work with:
  - ResNet50
  - ResNet152
  - YOLO

All these are implemented completely and were used to recognize multiple objects in a video:
  - Object_Recognition_ResNet50.py
  - Object_Recognition_ResNet152.py
  - YOLO.py
  - Yolo_Vid.py

A Train.py allows you to train your custom datasets on these models for your use & application.


ABSTRACT

Real-time object detection and tracking is a vast, vibrant yet inconclusive and complex area of computer vision.
Due to its increased utilization in surveillance, tracking system used in security and many others applications have propelled researchers 
to continuously devise more efficient and competitive algorithms. However, problems emerge in implementing object detection and tracking in 
real-time; such as tracking under dynamic environment, expensive computation to fit the real-time performance, or multi-camera multi-objects
tracking make this task strenuously difficult. Though, many methods and techniques have been developed, but in this literature review we 
have discussed some famous and basic methods of object detection and tracking. In the end we have also given their general applications and
results.


INTRODUCTION

 1.1 OBJECT RECOGNITION
 
What Is Object Recognition?

Object recognition is a computer vision technique for identifying objects in images or videos. 
Object recognition is a key output of deep learning and machine learning algorithms. When humans look at a photograph or watch a video, 
we can readily spot people, objects, scenes, and visual details. The goal is to teach a computer to do what comes naturally to humans: 
to gain a level of understanding of what an image contains.
Object recognition is a key technology behind driverless cars, enabling them to recognize a stop sign or to distinguish a pedestrian from
a lamppost. It is also useful in a variety of applications such as disease identification in bioimaging, industrial inspection, and robotic vision.

Object Recognition vs. Object Detection

Object detection and object recognition are similar techniques for identifying objects, but they vary in their execution. Object detection is the process of finding instances of objects in images. In the case of deep learning, object detection is a subset of object recognition, where the object is not only identified but also located in an image. This allows for multiple objects to be identified and located within the same image.

How Object Recognition Works

You can use a variety of approaches for object recognition. Recently, techniques in machine learning and deep learning have become popular approaches to object recognition problems. Both techniques learn to identify objects in images, but they differ in their execution.

1.2 RESIDUAL NUERAL NETWORK

A residual neural network (ResNet) is an artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in
the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers.
Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in be
tween. An additional weight matrix may be used to learn the skip weights; these models are known as HighwayNets. Models with several parallel
skips are referred to as DenseNets. In the context of residual neural networks, a non-residual network may be described as a plain network.

A reconstruction of a pyramidal cell. Soma and dendrites are labeled in red, axon arbor in blue. (1) Soma, (2) Basal dendrite, (3) Apical dendrite, (4) Axon, (5) Collateral axon.
One motivation for skipping over layers is to avoid the problem of vanishing gradients, by reusing activations from a previous layer until
the adjacent layer learns its weights. During training, the weights adapt to mute the upstream layer, and amplify the previously-skipped
layer. In the simplest case, only the weights for the adjacent layer's connection are adapted, with no explicit weights for the upstream
layer. This works best when a single nonlinear layer is stepped over, or when the intermediate layers are all linear. If not, then an 
explicit weight matrix should be learned for the skipped connection (a HighwayNet should be used).
Skipping effectively simplifies the network, using fewer layers in the initial training stages. This speeds learning by reducing the 
impact of vanishing gradients, as there are fewer layers to propagate through. The network then gradually restores the skipped layers as 
it learns the feature space. Towards the end of training, when all layers are expanded, it stays closer to the manifold and thus learns 
faster. A neural network without residual parts explores more of the feature space. This makes it more vulnerable to perturbations that 
