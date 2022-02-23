# CUGAN

### Toward Interactive Modulation for Photo-Realistic Image Restoration. [(Paper Link)](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Cai_Toward_Interactive_Modulation_for_Photo-Realistic_Image_Restoration_CVPRW_2021_paper.html)
By [Haoming Cai*](https://scholar.google.com/citations?user=mePn76IAAAAJ&hl=en), [Jingwen He*](https://scholar.google.com/citations?hl=en&user=GUxrycUAAAAJ&view_op=list_works&sortby=pubdate), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en) in CVPRW, NTIRE workshop 2021.

<p align="center"> 
  
  <img src="figures/modulation.png">
  Two-dimension Modulation
  
</p>

<p align="center">

  <img src="figures/modulation_real.png">
  Real-World Modulation

</p>

## Dependencies and Installation
- pip install -r requirements.txt
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


## How to Test
- **Prepare the test dataset**
	1. Download LIVE1 dataset and CBSD68 dataset from [Google Drive](https://drive.google.com/drive/folders/1-ye2s6og03jHh5A0cjtINpOUickJEra0?usp=sharing)
	1. Generate LQ images with different combinations of degradations using matlab [`codes/data_scripts/generate_2D_val.m`](codes/data_scripts/generate_2D_val.m).


- **Download the pretrained model**
	1. Download pretrained CUGAN from [Google Drive](https://drive.google.com/drive/folders/11NjU4ov7g6dU0DK5ldt43TIZKBmXFSi9?usp=sharing)
	1. Modify the `pretrain_model_G` in configuration file [`options/test/xxxxxx.yml`]().


- **Test CUGAN with range of restoration strength**
	1. Modify the configuration file [`options/test/modulation_CUGAN.yml`](codes/options/test/modulation_CUGAN.yml). ❗️Importantly, `cond_init`, `range_mode`, `range_stride` are crucial in this testing mode.
	1. Run command:
	```c++
	cd codes
	python test-cugan_range-cond.py -opt options/test/modulation_CUGAN.yml
	```
- **Test CUGAN with specific restoration strength**
	1. Modify the configuration file [`options/test/test_CUGAN.yml`](codes/options/test/test_CUGAN.yml). ❗️Importantly, `cond` is crucial in this testing mode.
	1. Run command:
	```c++
	python test-cugan_specific-cond.py -opt options/test/test_CUGAN.yml
	```

## How to Train
- **Cooming Soon**


## Acknowledgement

- This code is based on [mmsr](https://github.com/open-mmlab/mmsr) and [CResMD](https://github.com/hejingwenhejingwen/CResMD).