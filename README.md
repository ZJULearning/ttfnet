# TTFNet: 

This project hosts the code for implementing the TTFNet algorithm for object detection, as presented in our paper:

The full paper is available at: 

## Highlights
- **anchor-free:**  TTFNet .

## Required hardware
We use 8 Nvidia GTX 1080Ti GPUs.

## Installation
This TTFNet implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Therefore the installation is the same as original mmdetection.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MMDETECTION_README.md) of mmdetection.

## A quick demo
Once the installation is done, you can follow the below steps to run a quick demo.



## Inference
The inference command line on coco minival split:



For your convenience, we provide the following trained models.

**ResNets:**

*All ResNet based models are trained with xx images in a mini-batch.*

Model | Total training Hours | Testing time(fps) | AP (minival) | AP (test-dev) | Link

## Training

The following command line will train TTF_D53_2x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
```

## License

