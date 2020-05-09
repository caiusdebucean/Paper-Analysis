# CenterMask: Real-Time Anchor-Free Instance Segmentation
The following analysis is based on _CenterMask : Real-Time Anchor-Free Instance Segmentation_ paper:
 
* https://arxiv.org/abs/1911.06667
 
The code found on [Github](https://github.com/youngwanLEE/CenterMask).
 
___
 
## Introduction
 
CenterMask is a simpy and efficient anchor-free instance segmentation solution. With the addition of spacial attention-guided mask (SAG-Mask) branch to anchor-free one stage object detector _FCOS_. The SAG-Mask branch predicts a segmentation mask on each detected box with spatial attention maps, which focuses on informative pixels and suppresses noise.
Another big improvement is the introduction of VoVNetV2, which is a direct upgrade from the original VoVNet because of the optimization for larger variations and the use of effective Squeeze-Excitation.
 
## Architecture
 
The architecture consists of 3 modules: _Backbone + Feature Pyramid Network_, _FCOS Multi-level heads_, _SAG-Mask_.
 
<div style="text-align:center"><img src="https://i.imgur.com/VqBUH4D.png"/width = 100%></div>
 
The pipeline is quite simple: Image features are produced by the _**Backbone + FPN**_. Each level of the **_FPN_** feeds into separate _**FCOS BOX Heads**_ which regress different-sized boxes on each _FPN_ level. These Boxes are then fed into the _**SAG-Mask**_ to predict segmentation masks in a per-pixel manner.

### Backbone + FPN
 
 _To better understand this architecture module, read the analysis on FCOS: Fully-Convolutional One-Stage Object Detection._
 To sum this up, this module's architecture looks like this:

<div style="text-align:center"><img src="https://i.imgur.com/vGuZGxI.png" width = 100%><i>Photo credited to FCOS: Fully Convolutional One-Stage Object Detection paper</i></div>

<br />

The main improvement made from the original _FCOS_ architecture is the introduction of **VoVNetV2** backbone, which is a direct upgrade of **VoVNet**.
**VoVNet** uses *One-Shot Aggregation* modules (which we will call *OSA* Modules). Here is an illustration of the main difference of OSA modules:

<div style="text-align:center"><center><img src="https://i.imgur.com/xDw055g.png" width = 70%></center><i>Photo credited to An Energy and GPU-Computation Efficient Backbone Networkfor Real-Time Object Detection paper</i></div>

<br />

The improvements made to the original VoVNet are: 

1. Adding a **residual connection**, to solve saturation or degradation of deeper variations of VoVNet.
2. Using **Effective Squeeze-Excitation** channel attention modules. These are improved from the *original* **Squeeze-Excitation** modules, which squeeze the spacial dependency by global average pooling followed by two FC layers followed by a Sigmoid. These are used to rescale the input feature map and highlight only useful channels. Due to the original SE module's limitations of channel information losses due to dimension reduction, the **effective** improvement added in this paper proposes the use of only one FC layer without channel dimension reduction.

A good representation of these improvements:

<div style="text-align:center"><center><img src="https://i.imgur.com/Wtg7sRx.png" width = 1000%></center></div>

<br />
 
### FCOS Multi-level BOX Heads

_To better understand this box regressor module, read the analysis on FCOS: Fully-Convolutional One-Stage Object Detection._

To summarize, (x,y) locations represent pixels in a feature map that can be traced back to the input image. This means there are _H x W_ locations. For each location in each _FPN_ level, a set of 3 branches are predicted: an 80D class branch, a 1D centerness branch which predicts how far a location is from the ground-truth box center, and a 4D box regression branch, which predicts the 4 distances from the respective location to the bounding box: _left, right, top and bottom_. FCOS takes into consideration all pixels that are correctly classified in a ground-truth box as positive samples and does regression on them. If a location falls into multiple ground-truth boxes, the location is attributed to the smallest area bounding box.


<div style="text-align:center"><center><img src="https://i.imgur.com/IWmUbo7.png" width = 60%></center><i>Photo credited to FCOS: Fully Convolutional One-Stage Object Detection paper</i></div>

<br />



After object proposals are predicted, CenterMask predicts segmentation masks by also using **RoI Align**. The main difference between *RoI Pooling* and *RoI Align* is **quantization**. *RoI Align* is not using quantization for data-pooling. For example, Fast-R-CNN is applying quantization twice: first in the mapping process then second during the pooling process. *RoI Align* skips that by dividing the original RoI into equal boxes and applying bilinear interpolation inside every one of them. 

The comparison looks like this:

<div style="text-align:center"><center><img src="https://i.imgur.com/HggFg5j.png" width = 100%></center><i>Photo credited to Kemal Erdem's Understanding Region of Interest article on towardsdatascience.com</i></div>

<br />

CenterMask introduces an adaptation in RoI Align, by modifying the original Equation:
![Original](https://i.imgur.com/MJ6jf8H.png)
which is adapted to ImageNet pretraining size _224_, to:
![New](https://i.imgur.com/i18wqzd.png)

The variable _k_ represents the level associated with _(P3, P4, ... , P7)_. If _k is lower than P3, it's clamped to the minimum level. Alternatively, if an area of a RoI is bigger than half the input area, then RoI is assigned to the highest feature level (_e.g. P7_).

### Spatial Attention-Guided Mask

Before we dive into anything, attention methods are used to focus on important features, but also suppress unnecessary ones. Keep in mind that channel attention emphasizes on 'what' to focus across channels of feature maps, while spatial attention emphasizes on 'where' is an informative region.

Now, to calculate the **Spatial Attention Map** A<sub>sag</sub>, the features inside the predicted RoIs are extracted by RoI Alignt with 14x14 resolution, then they are fed into **four _conv_ layers** and then to a _**spatial attention module (SAM)**_. _SAM_ generates pooled features by **both average and max pooling** operations along the channel axis, and aggregates them via concatenation. That is followed by a **3x3 convolution** and *normalized* by a **sigmoid function**. 

<div style="text-align:center"><img src="https://i.imgur.com/8HQxmfs.png
" width = 50%></div>

Finally, the **Attention Guided Map** X<sub>sag</sub> is computed as:

<div style="text-align:center"><img src="https://i.imgur.com/o2M1T5Y.png
" width = 40%></div>

where ⊗ denotes element-wise multiplication. Then, a **2x2 deconvolution** upsamples the spacially attended feature map to 28x28 resolution. Lastly, a **1x1 convolution** is applied for predicting class-specific masks *(for all 80 classes)*.


## Loss function

The loss function is calculated as follows:

<div style="text-align:center"><img src="https://i.imgur.com/BS06ACf.png
" width = 40%></div>

with <b>L<sub>cls</sub>, L<sub>center</sub></b>, and <b>L<sub>box</sub></b> being the classification, centerness and box regression losses, same as in _FCOS_. <b>L<sub>mask</sub></b> is the average binary cross-entropy loss, same as in _Mask R-CNN_.


## Results

These are the results on *COCO testdev*:

<div style="text-align:center"><center><img src="https://i.imgur.com/LfEefsE.png" width = 100%></center></div>

<div style="text-align:center"><center><img src="https://i.imgur.com/2dZics3.png" width = 100%></center></div>



_All non-attributed photos are credited to the original paper (source presented above)._
 
<div>&copy; <b>Debucean Caius-Ioan</b> @ github.com/caiusdebucean</div>
<div> <i>First created on May 2020 during COVID-19 quarantine </i></div>