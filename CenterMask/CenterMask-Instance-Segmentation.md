# CenterMask: Real-Time Anchor-Free Instance Segmentation
The following analysis is based on _CenterMask : Real-Time Anchor-Free Instance Segmentation_ paper:
 
* https://arxiv.org/abs/1911.06667
 
The code found on [Github](https://github.com/youngwanLEE/CenterMask).
 
___
 
## Introduction
 
CenterMask is a simpy and efficient anchor-free instance segmentation solution. With the addition of spacial attention-guided mask (SAG-Mask) branch to anchor-free one stage object detector _FCOS_. The SAG-Mask branch predicts a segmentation mask on each detected box with spatial attention maps, which focuses on informative pixels and suppresses noise.
Another big improvement is the introduction of VoVNetV2, which is a direct upgrade from the original VoVNet because of the optimization for larger variations and the use of effectife Squeeze-Excitation.
 
## Architecture
 
The architecture consists of 4 modules: _Backbone + Feature Pyramid Network_, _FCOS Multi-level heads_, _SAG-Mask_.
 
<div style="text-align:center"><img src="https://i.imgur.com/VqBUH4D.pngg"/width = 100%></div>
 
### Backbone + FPN
 
 
 
 
_All photos are credited to the original paper (source presented above)_
 
<div>&copy; <b>Debucean Caius-Ioan</b> @ github.com/caiusdebucean</div>
<div> <i>First created on May 2020 during COVID-19 quarantine </i></div>