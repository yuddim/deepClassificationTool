# deepClassificationTool 
Deep Image Classification Tool based on Keras. Tool implements light versions of VGG, ResNet and InceptionV3 for small images.
Tool uses python 3.5. 

Tool has 3 modes:
1) Training of new deep neural network.
2) Tuning of existing deep neural network.
3) Testing of existing existing (trained) deep neural network.

For training and tune mode you need two folders:
1) Training folder with subfolders - one for each image class
2) Test folder with subfolders - one for each image class

Example of folders tree:

     train/ImageClass1
          /ImageClass2
          /ImageClass3
          /ImageClass4
          ...

    test/ImageClass1 
        /ImageClass2
        /ImageClass3
        /ImageClass4
        ...

Here each subfolder 'ImageClassi' consists set of images of i-th class.

Description of main modules:

deepClassificationTool.py - main module for training and testing of deep neural network

testFunctions.py - functions for testing of trained deep neural network on one image, two images and folder of images

modelVGGm.py - Light version of VGG for small images - Inspired from VGG, 2014 - VGGm(modified)

modelResNetM.py - Light version of ResNet for small images - resNetM (modified)

                    # Reference - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
                    
                    # Reference - [https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py]
                    
modelInceptionV3m.py - Light version of inceptionV3 - inceptionV3m (modified)

                    # Reference - [https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py]
                    
                    # Reference - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
