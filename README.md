# Residual-Attention-Network-for-Image-Classification-2D-3D-pytorch-implement
2D and 3D(volume) verison of Residual Attention Network

Residual Attention Network for Image Classification (CVPR-2017 Spotlight) By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang* (https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf)

## Use Examples:
  ### 2D:
  - `from attentionnet import attention56, attention92`
  - `attention56(num_classes=1000)`
  - `attention92(num_classes=1000)`
  ### 3D:
  - `from attentionnet3D import attention3d56, attention3d92`
  - `attention3d56(num_classes=1000)`
  - `attention3d92(num_classes=1000)`
  
  For the usage in a library, please refer to my fork on pretorched (https://github.com/moyiliyi/pretorched-x)
  
  Only the network architectures implemented here. You need to write your own train/test scripts. 

## Reference
This code is based on the following repos:
- https://github.com/weiaicunzai/pytorch-cifar100
- https://github.com/pytorch/vision/tree/master/torchvision
- https://github.com/Tencent/MedicalNet
