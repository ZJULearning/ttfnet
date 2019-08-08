# Training Time Friendly Network for Real-Time Object Detection 

The code for implementing the **TTFNet** for object detection.

![image-20190807160835333](imgs/structure.png)

## Highlights
- **Simple:** Anchor-free, no multi-stage, no multi-scale features, no time-consuming post-processing. TTFNet only requires two detection heads for object localization and size regression. Any other predictions used to correct the results are not required.
- **Training Time Friendly:**  Our fast-training TTFNet-18 and TTFNet-53 can achieve 26 AP / 111 FPS within 2 hours and 33 AP / 55 FPS within 4 hours on the MS COCO dataset using 8 GTX 1080Ti.
- **Fast and Precise:** Our TTFNet-18/34/53 can achieve 27.9AP / 112FPS, 31.3AP / 87FPS, and 35.1AP / 54.4 FPS on 1 GTX 1080Ti.

![image-20190807153122553](imgs/results.png)

## Installation
Our TTFNet is based on [mmdetection](https://github.com/open-mmlab/mmdetection). You may also want to see the original [README.md](MMDETECTION_README.md) of mmdetection. 

Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Inference
The inference command line on coco minival split:



We provide the following converged models.

| Model            | Training Hours | FPS   | AP(minival) | Link                                                         |
| ---------------- | -------------- | ----- | ----------- | ------------------------------------------------------------ |
| TTFNet-18 (fast) | 2.1            | 110.7 | 25.9        | [Download](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/EaCB-TKnqURNoyl_P-uxClABsSz35Nmu-L1T3SObBnHYMw?e=HxGXPE) |
| TTFNet-18        | 4.1            | 112.3 | 28.1        | [Download](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/Ef2CVhUYfOJDjzrlQfZGWxwBvUqnXu3mJ0pweiSEuPNYag?e=iJwPPv) |
| TTFNet-34        | 4.8            | 86.6  | 31.3        | [Download](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/Eb0Ab71mpsFBvBP2_GsJ4tUBRGM8NvAym9qZxPqJGtyWSA?e=6Y7BU7) |
| TTFNet-53 (fast) | 3.4            | 54.8  | 32.9        | [Download](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/EZVE_d6oR_VGrDD4IZS2ppIB0lm2V8UaBkgFUjuEM7oRZA?e=XVLRwx) |
| TTFNet-53        | 6.8            | 54.4  | 35.1        | [Download](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/Ed1Qxlom3FpGmNZI9dnHau8BtXF0rPeHSYGn6HkoXOEB3A?e=NfPjrZ) |

We also provide the pretrained [Darknet53](https://zjueducn-my.sharepoint.com/:u:/g/personal/11921047_zju_edu_cn/EaXXohf5LgBNji6bkxrARN4BZ9N4sEedaINPeqexu5l2jA?e=nJhe8L) here.

## Training

The following commands will train TTFNet-18 on 8 GPUs for 24 epochs and TTFNet-53 on 8 GPUs for 12 epochs:

```
./scripts/dist_train.sh configs/ttfnet/ttfnet_r18_2x.py 8
```

```
./scripts/dist_train.sh configs/ttfnet/ttfnet_d53_1x.py 8
```



## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```

```

## License

