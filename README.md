# Self-Supervised Class-Agnostic Motion Prediction with Spatial and Temporal Consistency Regularizations
Official implementation for our CVPR2024 paper: "Self-Supervised Class-Agnostic Motion Prediction with Spatial and Temporal Consistency Regularizations". [**[Arxiv]**](https://arxiv.org/pdf/2403.13261.pdf)

## 🔨 Dependencies and Installation
- Python 3.9
- Pytorch >= 2.0
```
# git clone this repository
git clone https://github.com/kwwcv/SelfMotion
cd SelfMotion
```
### Dataset
- Download the [nuScenes data](https://www.nuscenes.org/).
```
# modified the following paths in gen_data.py, gen_GSdata.py, and data_utils.py
# sys.path.append('root_path/SelfMotion')
# sys.path.append('root_path/SelfMotion/nuscenes-devkit/python-sdk/')
```
- Run command `python data/gen_data.py` to generate preprocessed BEV data for validating, and testing. Refer to [MotionNet](https://github.com/pxiangwu/MotionNet) and `python data/gen_data.py -h` for detailed instructions.
  
- Install the ground segmentation algorithm following [Patchwork++](https://github.com/url-kaist/patchwork-plusplus). One can also try removing the ground points by simply setting a threshold along the Z-axis.
```
# modified the following path in gen_GSdata.py
# patchwork_module_path = "root_path/patchwork-plusplus/build/python_wrapper"
```
- Run command `python data/gen_GSdata.py` to generate preprocessed ground-removed BEV data for training.
## 🔥 Training
```
python train.py --train_data [ground removal bev training folder] --test_data [bev validation folder] \
       --log --log_path [path to save log] --if_cluster --if_forward --if_reverse
```

## 🎯 Evaluation
[Download Pretrained Model](https://drive.google.com/file/d/1jQa6CB7K6UFUU-xPWoU9BZn7lXMMA_wV/view?usp=sharing)
```
python test.py --data [bev testing folder] --model [model path] \
      --log_path [path to save results]
```
## Citation
```
@misc{wang2024selfsupervised,
      title={Self-Supervised Class-Agnostic Motion Prediction with Spatial and Temporal Consistency Regularizations}, 
      author={Kewei Wang and Yizheng Wu and Jun Cen and Zhiyu Pan and Xingyi Li and Zhe Wang and Zhiguo Cao and Guosheng Lin},
      year={2024},
      eprint={2403.13261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🍭 Acknowledgement
Our project is based on
[MotionNet](https://github.com/pxiangwu/MotionNet)

The optimal transport solver is adopted from
[Self-Point-Flow](https://github.com/L1bra1/Self-Point-Flow)

### License
This project is licensed under [NTU S-Lab License 1.0](LICENSE) 
