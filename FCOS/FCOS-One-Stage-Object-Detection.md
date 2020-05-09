# FCOS: Fully Convolutional One-Stage Object Detection
The following analysis is based on _FCOS: Fully Convolutional One-Stage Object Detection_ paper:

* https://arxiv.org/abs/1904.01355

The code found on [Github](https://github.com/tianzhi0549/FCOS/).

___

## Introduction

**FCOS** is a fully convolutional one-stage object detector that solves object detection in a per-pixel prediction fashion. *(Similar to Semantic Segmentation)*
The main difference between this method and other traditional methods such as RetinaNet, SSD, YOLO, or R-CNN variations, is that its one-stage, which means that it does not require a Region Proposal Network. Moreover, it's shown that FCOS cand be used as an RPN in other two-stage solutions. Lacking an RPN usually means sacrificing accuracy for inference speed. 

Another difference is that this method does not require predefined anchor boxes, which results in fewer hyperparameters to tweak and less complicated computation. 
This method also introduces a concept called _Centerness_ and it has a dedicated branch that predicts the deviation of a pixel to the center of its corresponding bounding box.

## Architecture
 
The network is composed of 3 parts: a **Convolutional Backbone**, a **Feature Pyramid Network** and **Output Heads** that encode *Centerness*, *Classes* and *4D Box Regressors*.

![Architecture](https://i.imgur.com/8U3USja.png)

## FCOS Heads _(output)_

The final layer of the network predicts an **80D vector of classification labels** and a **4D box regression vector**. Parallel to the _Class_ vector, a **Center-ness** value is also given, for multi-level prediction improvement. To be noted that the network outputs _C (number of classes)_ binary classifiers, instead of a softmax output.

### Bounding Box

Let us define the groundtruth bounding boxes as:

<div style="text-align:center"><img src="https://i.imgur.com/5CvvsoP.png"/width = 30%></div> 

with (x⁰ᵢ, y⁰ᵢ) and (x¹ᵢ, y¹ᵢ) being the top-left and bottom-right corner and cᵢ being the class.

We will also define each location on the feature map as ***(x,y).*** 
Also, we will define **Fᵢ** as the feature map at layer _**i**_ of a backbone CNN and _**s**_ as the total stride until that layer. _(This means that s will be a power of 2)_

Considering these, we can map each location _(x,y)_ back to the original input as ___(floor(s/2) + x * s, floor(s/2) + y * s)___ . If a location falls into one of the ground-truth boxes and the class label _c*_ of the location is the class label of the ground-truth box, then the location is considered a *positive* sample. Otherwise it's considered a *negative* sample and _c*_ = 0 (background class). 

The box variables are represented by a 4D real vector __t* = (l*, t*, r*, b*)__. These 4 values represent the distances from the *(x,y)* location to the four sides of the bounding box: *left*, *top*, *right*, *bottom*.
These can be calculated as:

* __l* = x - x⁰ᵢ__
* __t* = y - y⁰ᵢ__
* __r* = x⁰ᵢ - x__
* __b* = y⁰ᵢ - y__
  
<div style="text-align:center"><img src="https://i.imgur.com/WysaNzu.png"/width = 100%></div> 


_It is mentioned that "FCOS can leverage as many foregrounds samples as possible to train the regressor". As far as my understanding stretches, this means that every pixel or location, that is a positive sample and is not part of multiple boxes, will regress the box. Those pixels that fall into multiple boxes will regress only the smallest area box, as seen in the second image above._

Rather than considering anchor boxes with a high IoU with ground-truth boxes as positive samples, FCOS takes into consideration all pixels inside a ground-truth box, including marginal pixels (or locations), and performs regression with them. This is one of the reasons that it outperforms its anchor-based counterparts, with considerably fewer parameters.

Moreover, since the regression targets are always positive, exp(x) is used to map any real number to (0,∞) on top of the regression branch.

### Classification

Each location can fall into multiple bounding boxes, meaning that it can have multiple classes assigned to it. However, because of ambiguity, it is preferred to have **only one class** assigned to **each location**. The issue is resolved by **picking the smallest area** bounding box the location falls into and assigning the respective class as the location's only class. 
As mentioned earlier, if a location does not fall into any ground-truth bounding box, it is considered a negative sample, thus being part of the *background class*. 

### Multi-level Prediction with FPN

Before diving into describing the _Center-ness_ branch, we first need to motivate the presence of *multiple-level heads* for prediction.

There are two possible issues to the proposed FCOS that can be solved with multi-level predictions with FPN: 

1. The **large stride** (comparing to the original feature maps) can result in **low BPR** _(Best Possible Recall)_. For anchor-based solutions, this can be solved by lowering the IoU thresholds when deciding if a sample is positive or negative. For FCOS, you may think that the BPR can be much lower than anchor-based detectors, as it is impossible to recall an object which no location on the final feature maps encodes due to a very large stride. However, it is empirically proven that even with a large stride, FCOS is able to get a *good BPR*. Moreover, **with multi-level FPN prediction**, the **BPR** can be **improved** further.

2. **Overlaps** in ground-truth boxes can cause intractable **ambiguity** _(e.g. Which bounding box should a location regress?)_ This ambiguity leads to decreased performance of FCN-based detectors. However, **multi-level prediction** can **solve this problem**, and sometimes even perform *better* than anchor-based solutions.

Thus, the chosen approach detects different levels of feature maps. Relative to the _architecture illustration_ above, five levels of feature maps are defined as *{P3, P4, P5, P6, P7}*, with *P3, P4* and *P5* being produced by the backbone's CNN's feature maps *C3, C4* and *C5*, followed by a *1x1 convolution with top-down connections*. *P6* and *P7* is produced by doing *2-strided convolutions* on their previous layer. As a result, we have feature levels *P3, P4, P5, P6* and *P7* with *strides 8, 16, 32, 64 and 128*, respectively.

Unlike anchor-based solutions which assign anchor boxes with different sizes to different feature levels, a limit is put in place at each level of the FPN, limiting the range of bounding box regressions. Specifically, the __4D vector (l*, t*, r*, b*)__ is calculated for **each location on all feature levels**. Next, if a location satisfies __max(l*, t*, r*, b*) > mᵢ__ or __max(l*, t*, r*, b*) < m<sub>i-1</sub>__, it is set as a negative sample and is not required to regress a bounding box. The variable m<sub>i</sub> is the maximum distance that a feature level i needs to regress. The variables <i>m<sub>2</sub>, m<sub>3</sub>, m<sub>4</sub>, m<sub>5</sub>, m<sub>6</sub></i> are set to 0, 64, 128, 256, 512 and infinity. Considering these values, different feature levels will regress different size ranges, without any intersection in box sizes. _(e.g. [0,64] range for P3, [64,128] range for P4 and so on.)_ 

This is a brilliant workaround to overlapping boxes, resulting in a considerably different size between remaining ones that still overlap. This big difference in size implies that the remaining overlapping boxes refer to completely different objects.
Since this difference between levels is in place, insteand of using the standard _exp(x)_, a trainable scalar <i>s<sub>i</sub></i> is introduced, thus making use of _exp(x*<sub>i</sub>)_ adjustment at each level <i>P<sub>i</sub></i>, which slightly improves detection performance.

### Center-ness

Even with multi-level predictions in FCOS, a performance gap still exists when compared to anchor-based detectors. Thus, a simple and effective strategy to suppress the low-quality detected bounding boxes without any new _hyperparameters_ is to add a single-layer parallel branch to the existing classification one. This new branch predicts the **center-ness** of a location. 

This **center-ness** is defined as the normalized distance from the location to the center of the object that the location is depicting.
Given the regression 4D vector, the center-ness is computed as:

<div style="text-align:center"><img src="https://i.imgur.com/48NOhcn.png"/width = 50%></div>

The _sqrt_ is employed to slow down the decay of the center-ness. Since its range is between 0 and 1, BCE loss is applied. This loss will be added to the **Loss function** described below. When testing, the final score is computed by multiplying the predicted center-ness with the corresponding classification score. 

As a consequence, center-ness can **down-weight** the scores of bounding boxes far from the center of the object. The low-quality bounding boxes might be filtered out by the **final NMS process**, improving the detection _remarkably_.

<div style="text-align:center"><img src="https://i.imgur.com/S47RHrJ.png"/width = 90%></div>


An alternative to _center-ness_ is to make use of only central portions of the ground-truth bounding boxes as positive samples with the price of one extra hyper-parameter. Apparently, the combination of both methods achieves a much better performance.

## Loss function

It is defined as follows:

<div style="text-align:center"><img src="https://i.imgur.com/Iq5DhuF.png"/width = 60%></div>

with L<sub>cls</sub> is the focal loss and L<sub>reg</sub> is the IoU loss. N<sub>pos</sub> denotes the number of positive samples and lambda being 1 in this paper is the balance weight for L<sub>reg</sub>. THis summation is calculated over all locations on the feature maps F<sub>i</sub>. _1<sub>{c<sub>i</sub><sup>*</sup> > 0}_ is the indicator function being 1 if c<sub>i</sub><sup>*</sup>  > 0 and 0 otherwise.

## Experiments and Results

The dataset used is COCO _trainval135k_ for training and _minival_ for validation. The main results are inferenced with *test_dev*.`

<div style="text-align:center"><img src="https://i.imgur.com/NMNCSYg.png"/width = 100%></div>



_All photos are credited to the original paper (source presented above)_

<div>&copy; <b>Debucean Caius-Ioan</b> @ github.com/caiusdebucean</div>
<div> <i>First created on May 2020 during COVID-19 quarantine </i></div>