# EDWL (ECCV2022)

An Gia Vien and Chul Lee

Official PyTorch Code for **"Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging"**

Paper link: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670429.pdf

### Introduction
![](/figs/Overview.png)

We propose a novel single-shot high dynamic range (HDR) imaging algorithm based on exposure-aware dynamic weighted learning, which reconstructs an HDR image from a spatially varying exposure (SVE) raw image. First, we recover poorly exposed pixels by developing a network that learns local dynamic filters to exploit local neighboring pixels across color channels. Second, we develop another network that combines only valid features in well-exposed regions by learning exposure-aware feature fusion. Third, we synthesize the raw radiance map by adaptively combining the outputs of the two networks that have different characteristics with complementary information. Finally, a full-color HDR image is obtained by interpolating missing color information.

### Requirements
- PyTorch 1.7.1/1.9.0
- Python 3.8.5/3.8.8
- mat73
- Matlab 2020a (for estimating evaluation metrics: pu-/log-PSNR, pu-MSSSIM, HDR-VDP, and HDR-VQM)

### Set up
- Test data path (e.g., "Kalantari/")
- Output path (e.g., "test_results/")
- Weight path (e.g., "WEIGHTS_ECCV2022/")

- Download Kalantari data for testing from: https://drive.google.com/file/d/1bkyNjlMst8rz5xRI43uzkOwhtNiTMWI2/view?usp=sharing
- Download pretrained weights from: https://drive.google.com/file/d/1v32KDb7qwck7lJL59m5ei7eGeGA6Qvjx/view?usp=sharing

### Usage
Running the test code:
```
    $ python Main_testing.py
```

###
**We are in preparing to share train and evaluation metric estimation codes soon!**

### Citation
Please cite the following paper if you feel this repository useful.
```
    @inproceedings{EDWL,
        author    = {An Gia Vien and Chul Lee}, 
        title     = {Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2022}
    }
```
### License
See [MIT License](https://github.com/viengiaan/EDWL/blob/main/LICENSE)
