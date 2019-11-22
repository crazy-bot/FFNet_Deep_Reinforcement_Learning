
This is pytorch implementation of the paper  [FFNet: Video Fast-Forwarding via Reinforcement Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lan_FFNet_Video_Fast-Forwarding_CVPR_2018_paper.pdf)

# FFNet_Implementation

FFNet_Tensor - all files of tensor
FFNet_Tensor/input.zip - feature and gt file from author
FFNet_Tensor/input2.zip - our feature and gt file
FFNet_Tensor/extract_feat.py - extract features using tensor. but weights bvlc_alexnet.npy need to be downloaded. it exceeds the size than permitted here.

FFNet_Torch - our implementation
FFNet_Torch/tour20_out/tour20_test_auth.png - coverage metric according to author
FFNet_Torch/tour20_out/tour20_test_me.png - coverage metric I understood
similar for tvsum50

FFNet_Torch/util.py - contains functions for data preparation
FFNet_Torch/video2frame.py - video to frames
other files are as per their naming.




