
This is pytorch implementation of the paper  [FFNet: Video Fast-Forwarding via Reinforcement Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lan_FFNet_Video_Fast-Forwarding_CVPR_2018_paper.pdf)

## FFNet_Implementation

### Official implementation can be found at [FFNET](https://github.com/shuyueL/FFNet)

### Our Implementation in Pytorch

**Dataset was downloaded as per the main github page**

Tour20 dataset:[Tour20](https://vcg.ece.ucr.edu/)

TVSum dataset:[TVSum](https://github.com/yalesong/tvsum)

**Pre and Pose processing of data (not present in official page)**

- video2frame.py
- util.py
- createVideo.py

#### Results ( mean coverage of keyframes & Hit Number )
- TVSum50
![tvsum50](https://github.com/crazy-bot/FFNet_Implementation/blob/master/FFNet_Torch/tvsum50_out/tvsum50_test_me.png)
- Tour20

![tour20](https://github.com/crazy-bot/FFNet_Implementation/blob/master/FFNet_Torch/tour20_out/tour20_test_me.png)








