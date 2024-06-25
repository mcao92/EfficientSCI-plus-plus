# Hybrid CNN-Transformer Architecturefor Efficient Large-ScaleVideo Snapshot CompressiveImaging (IJCV 2024)
Miao Cao, Lishun Wang, Mingyu Zhu and Xin Yuan

<hr />

> **Abstract:** Video snapshot compressive imaging (SCI) uses a low-speed 2D detector to capture high-speed scene, where the dynamic scene is modulated by different masks and then compressed into a snapshot measurement. Following this, a reconstruction algorithm is needed to reconstruct the high-speed video frames. Although state-of-the-art (SOTA) deep learning-based  reconstruction algorithms have achieved impressive results, they still face the following challenges due to excessive model complexity and GPU memory limitations: 1) these models need high computational cost, and 2) they are usually unable to reconstruct large-scale video frames at high compression ratios. To address these issues, we develop an efficient network for video SCI by using hierarchical residual-like connections and hybrid CNN-Transformer structure within a single residual block, dubbed EfficientSCI++. The EfficientSCI++ network can well explore spatial-temporal correlation using convolution in the spatial domain and Transformer in the temporal domain, respectively. We are the first time to demonstrate that a UHD color video ($1644\times{3840}\times{3}$) with high compression ratio ($40$) can be reconstructed from a snapshot 2D measurement using a single end-to-end deep learning model with PSNR above 34 dB. Moreover, a mixed-precision model is trained to further accelerate the video SCI reconstruction process and save memory footprint. Extensive results on both simulation and real data demonstrate that, compared with precious SOTA methods, our proposed EfficientSCI++ and EfficientSCI can achieve comparable reconstruction quality with much cheaper computational cost and better real-time performance.
<hr />

## Installation
Please see the [Installation Manual](docs/install.md) for EfficientSCI++ Installation. 

## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/EfficientSCI_plus_plus/efficientsci_plus_plus_base.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/EfficientSCI_plus_plus/efficientsci_plus_plus_base.py
```

## Testing EfficientSCI++ on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/EfficientSCI_plus_plus/efficientsci_plus_plus_base.py --weights=checkpoints/efficientsci_plus_plus_base.pth
```

Please contact me via caomiao92@gmail.com for the real testing datasets with continuous compression ratio ranging from 10 to 50.

## Citation

```
@article{cao2024hybrid,
  title={Hybrid CNN-Transformer Architecture for Efficient Large-Scale Video Snapshot Compressive Imaging},
  author={Cao, Miao and Wang, Lishun and Zhu, Mingyu and Yuan, Xin},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2024},
  publisher={Springer}
}
```
