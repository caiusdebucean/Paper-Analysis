# How do masks work in Yolact and how can we improve them for smaller objects
The following analysis is based on _Yolact and Yolact++: Real-time Instance Segmentation_ papers:
1. https://arxiv.org/abs/1904.02689
2. https://arxiv.org/abs/1912.06218
And on the code found at https://github.com/dbolya/yolact.

### There are 2 options to generate masks in the code:
In the  *config.py* file the class Config has 2 variables:
>'direct': 0

This produces masks directly as the output of each pred module - denoted as **fc-mask** in the paper.
>'lincomb': 1

Lincomb produces coefficients as the output of each pred module then uses those coefficients to linearly combine features from a prototype network to create image-sized masks. **This is the method they use**.
<div style="text-align:center"><img src="https://i.imgur.com/2mgVM4P.png" /></div> 

I will present the pipeline an image goes through. The **input** is an RGB image.
### Feature backbone
This takes the image in and runs it through the backbone chosen.
**Important**: More robust masks require deeper backbone features. This results in higher quality and ***better performance with small objects***. This is one of the reasons _ResNet_ is being chosen as the backbone (or any other modern Fully Convolutional Network). The other reason is that the task needs translation variance, and in the paper, it is explained that ResNet is inherently translation variant. 
### Feature Pyramid network
This takes the layers specified in the config.py file, mainly in:
>cfg.backbone.selected_layers

Then it adds downsampling layers and a layer is as the **Protonet** input through the config variable:
>cfg.mask_proto_src

The FPN presents the following output:
<div style="text-align:center"><img src="https://i.imgur.com/j9OMK4a.png" /></div> 

There are 3 coefficients as outputs: **class c, 4 bounding box regressors, k mask coefficients(corresponding to each prototype)**. This means that for every anchor (instance of an object) we produce **4 + c + k** per anchor. A smooth-L1 loss is used to train the box regressors and encode box regression coordinates in the same way as SSD. We apply tanh over **_k_** to be able to subtract them from one another. To tweak and edit this **_k_** parameter, check:
>yolact.py - line 425

>config.py - line 693

>functions.py - line 209

>functions.py - lines 168 - 187 - 200

>**_It is useful to check and calculate the input presented in config.py separately, in order to understand how it is used in the rest of the code._**

### Protonet
This takes the layer from the **FPN** and creates **_prototypes_**. We can also attach to the input layer a grid, to add grid features, but it is not specified what they are useful for. The Protonet has the following architecture:
<div style="text-align:center"><img src="https://i.imgur.com/pDJucl0.png" /></div> 

The **_prototypes_** are extracted from the last layer of it. The number of them is **_k_**, which is described above.
These **_prototypes_** look like this:
<div style="text-align:center"><img src="https://i.imgur.com/YImmZoQ.png" /></div> 

Or like this, for understaing their behaviour and how they **highlight** objects:
<div style="text-align:center"><img src="https://i.imgur.com/RG1jpbo.png" /></div> 

As you can observe, there are some invisible partitions that are in place. This is an Emergent behaviour of Yolact, which means they did not particularly code it, but it turned out to behave that way. In these examples, there are 4 and 6 respectively **_prototypes_**, which means k = 4 and k = 6.
In Yolact, they use 32. These are extracted from the last layer of the Protonet. 
To be noted that there is no "local" loss for the Protonet. The loss from the final assembly is used. Also, the output of the Protonet is unbounded, which means that it is not standardised/normalised, therefore the ReLU activation function is attached to the output.
### Prediction Head and Refinement
From the coefficients branch, a NMS function is used (they offer a refurbished faster NMS function). The code implementation for this is found at:
>detections.py - line 10, 81

At test time, this implementation is the final layer of the SSD-like FPN. It decodes location predictions, applies non-maximum suppression to location predictions based on the _k_-coefficient scores which are thresholded to be confident, and threshold to a _top_k_ number of output predictions for both confidence scores and locations, as the predicted masks.  
The **final masks _M_** are produced using a single matrix multiplication and sigmoid:
<div style="text-align:center"><i>M = sigmoid(PC<sup>T</sup>)

</i></div> 

where _P_ is an _h x w x k_ matrix of prototype masks and _C_ is a _n x k_ matrix of mask coefficients for *top_k* instances surviving NMS and score thresholding.

<div style="text-align:center"><img src="https://i.imgur.com/XUGkQoh.png" /></div> 

**Cropping** is a method used on the final masks. They are cropped according to the _predicted bounding box_ during evaluation and to the _groundtruth bounding box_ during training, for loss computation purposes.

**Leakage** means that the masks leak into other objects. Because the masks are cropped after assembly, and there is no attempt to suppress noise outside of the cropped region. This all works fine when the bounding box is accurate, but when it is not, that noise can creep into the instance mask and create some _leakage_ from outside the cropped region. This can also happen when two instances are far away from each other because the network has learned that it doesn't need to localize far away instances, as the cropping process will take care of it. This means that if the predicted bounding box is too big, the mask will include some of the far-away other instance's mask as well.

**Important observation:** Due to _VRAM_ limitations, only a subset of masks are used for training and calculating losses. In the code, they are selected in:
>config.py - 'masks_to_train':100              - line 461

They are sampled at random, which means that a big number of redundant and repetitive masks will only dilute the complexity of small objects the model will detect. 

### Conclusion and future experimentation
To go over some of the information presented above, we pointed out that the deeper backbone features we have, the better yolact will predict higher quality masks and perform better on **_small objects_**. Therefore, we can assume that the model is inherently performing better on _bigger objects_, and struggles more with _smaller objects_. To better perform on smaller objects, it is rational to think that the bigger number of **prototypes** we use, the better it will be at accurately predicting _smaller objects_. Although the paper results present that there is no increase in the _AP_ metric, as seen in this table:
<div style="text-align:center"><img src="https://i.imgur.com/wRswy1k.png" /></div> 

these were tested on the _COCO val2017_ dataset. Supposing that the goal of using Yolact is to detect smaller objects, it quickly becomes clear why the dataset used for benchmarking is not the most relevant to the task.
However, in the paper, it is mentioned that increasing _k_ can prove ineffective because predicting coefficients is difficult. If the network makes a large error in even one Coefficient, due to the nature of linear combinations in use, the produced mask can vanish or include **leakage** from other objects. 
Therefore, a balancing act needs to be played, as the right coefficients need to be produced, which means there is more space for error. In fact, it is mentioned that by **making _k_ a higher value**, the network simply **adds** redundant prototypes with **small edge-level variations** that _slightly increase AP_, which is something that may actually be on point with the task of detecting smaller objects, supposing they are not of complex shapes.

Also, to avoid **leakage**, there may be attempts to try and reduce the bounding box sizes in order to avoid errors at smaller objects, but this comes at the risk of cutting away from an overly correct detected object.

Another proposition comes to the assumption that we want to detect small objects on high-resolution frames. Supposing enough computation power is available, *segmenting* the frame is an option, with the risk of having a segmentation edge over an object. An alternative would be to **crop the input image to an AoI** (Area of Interest), thus making the small objects "bigger" in reference to total resolution.

<div>*<i>All the code indications are made on a clean Yolact repository at the date of creation</i></div>
<div>&copy;Debucean Caius-Ioan @ github.com/caiusdebucean</div>
<div> <i>First created on March 2020 during COVID-19 quarantine </i></div>
