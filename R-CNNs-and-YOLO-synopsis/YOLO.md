# YOLO synopsis
_This synopsis is based on the YOLO algorithm, found at:_
* https://arxiv.org/abs/1506.02640
* https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection

## Improvements from previous R-CNNs

Compared to other R-CNNs, where there are two outputs, and the loss is a weighted combination of classification and regression losses, the YOLO algorithm merges the two outputs into **a single one**.
So for example, if we have 3 classes, the **output** would look like this:

> y = [Pc, class1, class2, class3, x_center, y_center, width, height]

with _Pc_ being the probability that an object is **present** in that window. 

Considering the sliding window approach that's used by R-CNNs, we can propagate and calibrate a window through its feature maps, even when a size reduction is applied. The following concept can be observed below.

<div style="text-align:center"><img src="https://i.imgur.com/DPtN3ty.png" /></div> 

Because the sliding window approach can be slow, so choosing a stride so that the windows do not overlap and cover the whole image. The YOLO algorithm uses **grids** instead of sliding windows. Each **grid** cell will have an associated vector, containing the _output_ described above. This means that the output of the CNN will have the shape of _(grid_height, grid width, 1 + classes + 4)_. If a cell does not contain an object, it is discarded later. 

<div style="text-align:center"><img src="https://i.imgur.com/STD3a4P.png" /><i>Photo broken down in a 7x10 grid</i></div> 

## Training on grids

The question here is: _How does YOLO find a correct Bounding Box whet it looks at a grid-broken image?_ The answer is that YOLO assigns the groundtruth bounding box to only **one grid cell** in the training image. That means that only one cell can correctly locate an object during training. The way to figure out which grid cell is the correct one, YOLO first takes the **Groundtruth box** and applies it over the grid. Then, it calculates the ** centre point** of the groundtruth box. This means that the **correct grid cell is the one containing the centre point**. 


| <div style="text-align:center">Locating the correct grid cell</div>  | <div style="text-align:center">Calculating the _x_ and _y_ coordinates</div> |
| ------------- | ------------- |
| <div style="text-align:center"><img src="https://i.imgur.com/prvKoE6.png" width = 87%/></div> | <div style="text-align:center"><img src="https://i.imgur.com/B2uEOUG.png" width = /></div>   |

>As seen above, the _x_ and _y_ coordinates are relative to the grid cell, with the left top corner being the origin _(0,0)_ and the right bottom corner being _(1,1)_. Also, the width and height of the bounding box are relative to the image's width and height. This means that all 4 coordinates which form the bounding box around the object during training have values between _(0,1)_, which facilitates the training process. 

## Detecting Test Bounding Boxes

During testing time, the algorithm will detect many slightly different bounding boxes for the same object. To account for this, **Non-Maximal Suppression** is used, which tries to find the best bounding box in a group of proposed bounding boxes. 

To do **_NMS_** on a group of bounding boxes, we need to take into account IoU, which stands for _Intersection over Union_. Its value is $\frac{Area-of-Intersection}{Area-of-Union-of-Both-Boxes}$. This means for 2 bounding boxes to overlap, the IoU should be closer to 1, and if they do not overlap at all, their intersection is 0.

**_NMS_** selects the highest _Pc_ value - which is the probability that an object is present inside that bounding box - and then compares the IoU of other bounding boxes and the bounding box with the maximal _Pc_ value. If the IoU is bigger than a certain _threshold_, this means that the two bounding boxes might be trying to locate the same object, instead of locating two different close objects, so the algorithm gets rid of these bounding boxes.

<div style="text-align:center"><img src="https://i.imgur.com/rkUGtyQ.png" /><i>In this case, the bounding box with the highest Pc is the red one, and because the other bounding boxes have a high enough IoU with the red bounding box, they will be discarded</i></div> 

## Anchor Boxes and Multiple Objects Detection

If we have two overlapping objects, with the groundtruth centre located in the same grid cell, **Anchor Boxes** are a way to solve this problem, therefore allowing a grid cell containing multiple objects. **Anchor boxes** represent predefined bounding boxes for objects with a specific ratio. These are typically based on the sizes in the training dataset.

That means that a chosen number of _Anchor boxes_, The output vector for each _grid cell_ is modified to contain all _Anchor boxes_.

<div style="text-align:center"><img src="https://i.imgur.com/yfd46nT.png" /><i>In this example, we define two anchor boxes, and each grid cell will have the output shape of <b>y</b></i></div>

The main drawback to this is the case when two objects of the same shape, so they fit the same anchor box, overlap and have the groundtruth centre in the same cell.

## Algorithm summary

To summarize the algorithm, YOLO breaks up the image into a grid, then the CNN produces an output vector, size-dependent on the _number of classes_ and _number of existing anchor boxes_, for each grid cell. _NMS_ then gets rid of the bounding boxes that have a _Pc_ value lower than a certain threshold. Afterwards, it selects the bounding boxes with the highest _Pc_ value, and removes, based on IoU, the bounding boxes that are too similar to the _high-Pc_ bounding boxes.

___
<div>*<i>The information structurality is inspired by the Udacity CV Nanodegree, where the images are taken from.</i></div>
<div>&copy;Debucean Caius-Ioan @ github.com/caiusdebucean</div>
