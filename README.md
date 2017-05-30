# CIFAR-10 Classifier
This repository contains CIFAR-10 classifier using

1. MLP
2. CNN

MLP achives a 53% accuracy while CNN achieves a 85% accurcay. Implemented in Keras with a Tensorflow backend. There are two sets of pre-trained weights for each.

The files perform the following function
1. cifar_classifier.py - conatins models and training algorithms
2. cifar_predictor.py - uses stored weights , picks 32 images at random and creates jpegs with "value(prediction)" on top of numbers , the jpegs are essentially visualized predictions

pre-trained weights with **orig** tag in their file name , were trained on nvidia gtx 1080 GPU based on the models.

Required packages can be found in requirements.txt

The architecture uses augmentated data set in terms of position and rotations, It also uses increasing layers of filters and filter sizes, In accordance to starting with smaller features to bigger features. We do not go too big as well to avoid loosing generalization (overfitting).
