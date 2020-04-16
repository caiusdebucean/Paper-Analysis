# R-CNN variations synopsis
_This synopsis is based on the R-CNN/Fast R-CNN/Faster R-CNN algorithms, found at:_
* https://arxiv.org/abs/1311.2524
* https://arxiv.org/abs/1504.08083
* https://arxiv.org/abs/1506.01497
## R-CNN _(original)_

We use a region proposal algorithm to produce a limited set of cropped regions in an image. These are called **Regions of Interests _(RoIs)_**. Since the RoIs are of varying sizes, we need to **standardize** them to a chosen size. 
The R-CNN outputs a class score (this is a _classification_ task) and bounding box (_bb_) coordinates - _x,y_ for the center of the bb, width _w_ and height _h_ of the bb - for every input RoI (this is a _regression_ task).
**Observation**: Unlike the variations of R-CNNs, the original architecture does not explicitly produce a confidence score to detect whether an object is present in a RoI. Instead, it employs an approach of detecting the probability of a RoI being background. So in the end, R-CNN decieds if a RoI is a class object, or background.

<div style="text-align:center"><img src="https://i.imgur.com/DMuIwai.png"/></div>

## Fast R-CNN

This architecture runs the image through the CNN only **once**. A the end of the CNN we get a stack of feature maps. Since we still need to identify RoIs, we don't crop the original image, instead, we project the proposals on the feature maps layers. Each region in the feature map corresponds to a bigger region in the original image. However, RoIs need to be the same size, so it does **RoI pooling** to resize the images into a consistent size. Afterward, we put the selected RoIs through a _fc_ layer and we output the class and bb coordinates.

<div style="text-align:center"><img src="https://i.imgur.com/iCESs09.png"/></div>

This method is faster, as it only runs through the CNN once, but overall is still slow, as its test time is dominated by the time required to create region proposals.
_Explanation:_ **RoI pooling** takes in a rectangular region of any size and performs _maxpooling_ on that region in pieces such that the output is a fixed shape. The animation below is credited to _deepsense.ai_.
<div style="text-align:center"><img src="https://media.giphy.com/media/Tk8CNxVy5IfLWL7Xv0/giphy.gif"></img></div>

## Faster R-CNN

This architecture is similar to the rest, but it focuses on decreasing the time it takes to form region proposals. The images are still running through all the CNN until we have feature maps, but this time it looks at the produced feature map and takes a sliding window approach for binary classification on the presence of an object in that window.
Faster R-CNN uses a set of varying-size defined **anchor boxes** (_predefined boxes e.g. wide and short or tall and thin_). These are used to generate possible RoIs or _proposals_. For each proposal, there is a binary classification that detects if an object is present in that region. If a region has a small probability of containing an object, it is discarded. Those that have a bigger probability are runned through the classificator and box regressor.

<div style="text-align:center"><img src="https://i.imgur.com/zB7IDS1.png"/></div>

The speed of Faster R-CNN is decreased compared to the others because it reduces the time it takes to generate and decide on region proposals. An improved method that works in real-time is called YOLO, which gets rid of region proposals per se, and uses a predefined grid.








___
<div>*<i>The information structurality is inspired by the Udacity CV Nanodegree, where the most images are taken from. </i></div>
<div>&copy;Debucean Caius-Ioan @ github.com/caiusdebucean</div>
