# Video Object Recognition with Bounding Box Localization
Localize and identify classes in a video stream using a custom Resnet trained on PascalVOC 2007.  Uses Pytorch, Resnet and OpenCV.<br>
Note: This module only does prediction, and its only done on the CPU.  For training see the pascal.ipynb in the fastai Deep
Learning 2 course.<br>
Custom resnet34 has last fully connected and average pooling layers removed and a new head inserted consisting of two Fully Connected layers and their associated
companion bits (RELU, dropout).  The final FC layer has no softmax or sigmoid processing, just raw numbers.  The first 4 correspong to the x,y coordinates of top left and bottom right
positions of a bounding box.  The next 21 outputs correspond to each of the possible categories in PascalVOC (see cats.json for list).<br>
Since their is no processing you cannot tell how certain the network is of object identification.  To eliminate weak predictions, this model only shows predictions that are 3 standard
deviations outside the mean of all 21 predictions.  The final result and its bounding box are displayed like so:<br><br>
<p align="center">
<img src="https://github.com/kperkins411/video_object_recognition/blob/master/feltonID.png" width="300" height="300" />
</p>

## Run python files in this order
* video_capture4.py - does it all

## Problems
1. Item Can't find video camera: if seperate webcam it might be because its installed as root,first find it and see who owns it<br>
   ls -la /dev/video*<br>
   If root owns it then change ownership to you (assumming you are logged in as keith)<br>
    sudo chown -R keith /dev/video0 to place video0<br>
2. Item Can't find reg1.h5.  Yep this is a problem, its 110 Megs which exceeds github limits.  Please generate this file using the pascal.ipynb in the fastai Deep
Learning 2 course.

## Other files
utils.py - utility functions


