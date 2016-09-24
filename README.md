# About this fork

This Caffe fork was created on April 18, 2016. The changes to the original are listed below.

### 1. Weighted Euclidean (L2) loss 

The loss function computed in *EuclideanLossLayer* is changed to weighted Euclidean loss, which enables coordinate specific loss calculation.

[![ui](http://www.ee.oulu.fi/~malinna/images/posenet/weighted_euclidean_L2_loss.gif)](http://www.ee.oulu.fi/~malinna/images/posenet/weighted_euclidean_L2_loss.gif)

### 2. Multi-label support

Changed the *ImageDataLayer* to support multiple labels so that it can be used for regression tasks. The input file supports the following format, where the values can be, for example, coordinates (x1, y1, x2, y2, ....)

    01594.jpg 0.3284 0.8941 0.5021 0.7479 0.4534 0.5106
    01741.jpg 0.7152 0.9104 0.6800 0.7024 0.6288 0.5456
    01320.jpg 0.1612 0.4908 0.2821 0.5348 0.4835 0.3700

### 3. New accuracy layer

Added *PosenetAccuracyLayer* for logging accuracies and errors of coordinate predictions (xy) while training.

### 4. Faster R-CNN

Integrated Faster R-CNN code so that this fork supports Faster R-CNN for object detection.

### Instructions

Run `setup_posenet.m` first to build Faster R-CNN and download models.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
