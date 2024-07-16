Multi-branch Collaborative Learning Network for 3D Visual Grounding
=========
:tada::tada::tada:
This is a PyTorch implementation of MCLN proposed by our paper ["Multi-branch Collaborative Learning Network for 3D Visual Grounding"].**(ECCV2024)**
![image](https://github.com/qzp2018/MCLN/blob/main/data/fig.png)
## 0. Installation

+ **(1)** Install environment:
    ```
    conda create -n mcln python=3.7
    conda activate mcln
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install numpy ipython psutil traitlets transformers termcolor ipdb scipy tensorboardX h5py wandb plyfile tabulate einops
    ```
+ **(2)** Install spacy for text parsing
  ```
  pip install spacy
  # 3.3.0
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
  ```
+ **(3)** Compile pointnet++
  ```
  cd ~/MCLN
  sh init.sh
  ```
+ **(4)** Install segmentator from https://github.com/Karbo123/segmentator


## 1. Quick visualization demo 
We showing visualization via wandb for superpoints, kps points, bad case analyse, predict/ground_truth masks and box.
+ superpoints in 'src/joint_det_dataset.py' line 71
```
self.visualization_superpoint = False
```
+ others in 'src/grounding_evaluator.py' line 67 ~ 71
```
self.visualization_pred = False
self.visualization_gt = False
self.bad_case_visualization = False
self.kps_points_visualization = False
self.bad_case_threshold = 0.15
```

## 2. Data preparation

The final required files are as follows:
```
├── [DATA_ROOT]
│	├── [1] train_v3scans.pkl # Packaged ScanNet training set
│	├── [2] val_v3scans.pkl   # Packaged ScanNet validation set
│	├── [3] ScanRefer/        # ScanRefer utterance data
│	│	│	├── ScanRefer_filtered_train.json
│	│	│	├── ScanRefer_filtered_val.json
│	│	│	└── ...
│	├── [4] ReferIt3D/        # NR3D/SR3D utterance data
│	│	│	├── nr3d.csv
│	│	│	├── sr3d.csv
│	│	│	└── ...
│	├── [5] group_free_pred_bboxes/  # detected boxes (optional)
│	├── [6] gf_detector_l6o256.pth   # pointnet++ checkpoint (optional)
│	├── [7] roberta-base/     # roberta pretrained language model
│	├── [8] checkpoints/      # mcln pretrained models
```

+ **[1] [2] Prepare ScanNet Point Clouds Data**
  + **1)** Download ScanNet v2 data. Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission, and you will get the official download script `download-scannet.py`. Then use the following command to download the necessary files:
    ```
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.labels.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type .aggregation.json
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.0.010000.segs.json
    python2 download-scannet.py -o [SCANNET_PATH] --type .txt
    ```
    where `[SCANNET_PATH]` is the output folder. The scannet dataset structure should look like below:
    ```
    ├── [SCANNET_PATH]
    │   ├── scans
    │   │   ├── scene0000_00
    │   │   │   ├── scene0000_00.txt
    │   │   │   ├── scene0000_00.aggregation.json
    │   │   │   ├── scene0000_00_vh_clean_2.ply
    │   │   │   ├── scene0000_00_vh_clean_2.labels.ply
    │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene.......
    ```
  + **2)** Package the above files into two .pkl files(`train_v3scans.pkl` and `val_v3scans.pkl`):
    ```
    python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
    ```
+ **[3] ScanRefer**: Download ScanRefer annotations following the instructions [HERE](https://github.com/daveredrum/ScanRefer). Unzip inside `[DATA_ROOT]`.
+ **[4] ReferIt3D**: Download ReferIt3D annotations following the instructions [HERE](https://github.com/referit3d/referit3d). Unzip inside `[DATA_ROOT]`.
+ **[5] group_free_pred_bboxes**: Download [object detector's outputs](https://drive.google.com/drive/folders/1vfOeTLKdW2AFoQPoivxT5sFloeZSXnEf). Unzip inside `[DATA_ROOT]`. (not used in single-stage method)
+ **[6] gf_detector_l6o256.pth**: Download PointNet++ [checkpoint](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8) into `[DATA_ROOT]`.
+ **[7] roberta-base**: Download the roberta pytorch model:
  ```
  cd [DATA_ROOT]
  git clone https://huggingface.co/roberta-base
  cd roberta-base
  rm -rf pytorch_model.bin
  wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
  ```
+ **[8] checkpoints**: Our pre-trained models (see 3. Models).
+ **[9] ScanNetv2**: Prepare the preporcessed ScanNetv2 dataset follow "Data Preparation" section from https://github.com/sunjiahao1999/SPFormer, obtaining the dataset file with the following structure:
```
ScanNetv2
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
+ **[10] superpoints**: Prepare superpoints for each scene preprocessed from Step. 9.
  ```
  cd [DATA_ROOT]
  python superpoint_maker.py  # modify data_root & split
  ```

## 3. Models

|Dataset/Model  | REC mAP@0.25 | REC mAP@0.5 | RES mIoU | Model |
|:---:|:---:|:---:|:---:|:---:|
|ScanRefer/mcln| 57.17 |45.53 | 44.72 |[GoogleDrive](https://drive.google.com/file/d/1oBUWrTEj3kYyx-DT0HAvAcDUQe4nQgYz)


## 4. Training

+ Please specify the paths of `--data_root`, `--log_dir`, `--pp_checkpoint` in the `train_*.sh` script first.
+ Before Training and Evaluation, it's recommended to save pre-processed language features, in 'src/joint_det_dataset.py' line 135 ~ 140, which can save quite a lot of time.
+ For **ScanRefer** training
  ```
  sh scripts/train_scanrefer_mcln_sp.sh
  ```
+ For **ScanRefer (single stage)** training
  ```
  sh scripts/train_scanrefer_mcln_sp_single.sh
  ```
+ For **SR3D** training
  ```
  sh scripts/train_sr3d_mcln_sp.sh
  ```
+ For **NR3D** training
  ```
  sh scripts/train_nr3d_mcln_sp.sh
  ```

## 5. Evaluation

+ Please specify the paths of `--data_root`, `--log_dir`, `--checkpoint_path` in the `test_*.sh` script first.
+ For **ScanRefer** evaluation
  ```
  sh scripts/test_scanrefer_mcln_sp.sh
  ```
+ For **ScanRefer (single stage)** evaluation
  ```
  sh scripts/test_scanrefer_mcln_sp_single.sh
  ```
+ For **SR3D** evaluation
  ```
  sh scripts/test_sr3d_mcln_sp.sh
  ```
+ For **NR3D** evaluation
  ```
  sh scripts/test_nr3d_mcln_sp.sh
  ```

## 6. Acknowledgements

This repository is built on reusing codes of [EDA](https://github.com/yanmin-wu/EDA) and [3DRefTR](https://github.com/Leon1207/3DRefTR). We recommend using their code repository in your research and reading the [related article](https://arxiv.org/pdf/2209.14941.pdf). We are also quite grateful for [SPFormer](https://github.com/sunjiahao1999/SPFormer), [BUTD-DETR](https://github.com/nickgkan/butd_detr), [GroupFree](https://github.com/zeliu98/Group-Free-3D), [ScanRefer](https://github.com/daveredrum/ScanRefer), and [SceneGraphParser](https://github.com/vacancy/SceneGraphParser).

## 7. Citation

If you find our work useful in your research, please consider citing:
```
@misc{qian2024multibranchcollaborativelearningnetwork,
      title={Multi-branch Collaborative Learning Network for 3D Visual Grounding}, 
      author={Zhipeng Qian and Yiwei Ma and Zhekai Lin and Jiayi Ji and Xiawu Zheng and Xiaoshuai Sun and Rongrong Ji},
      year={2024},
      eprint={2407.05363},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.05363}}
```
