# CLIP-ViP (ICLR 2023)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-activitynet)](https://paperswithcode.com/sota/video-retrieval-on-activitynet?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/video-retrieval-on-lsmdc?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=clip-vip-adapting-pre-trained-image-text)


By [Hongwei Xue](https://hellwayxue.github.io/)\*, [Yuchong Sun](https://scholar.google.com/citations?user=DuSxNqgAAAAJ&hl=en)\*, [Bei Liu](https://www.microsoft.com/en-us/research/people/libei/), [Jianlong Fu](https://www.microsoft.com/en-us/research/people/jianf/), [Ruihua Song](https://scholar.google.com/citations?user=v5LctN8AAAAJ&hl=en), [Houqiang Li](http://staff.ustc.edu.cn/~lihq/en/), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/).


This repo is the official pytorch implementation of [CLIP-ViP: Adapting Image-Text Pre-training to Video-Language Representation Learning](https://arxiv.org/abs/2209.06430), accepted by [ICLR 2023](https://iclr.cc/Conferences/2023). CLIP-ViP is a video-language model which is based on a pre-trained image-text model [CLIP](https://openai.com/blog/clip/) then further pre-trained (post-pretraining) on a large-scale video-text dataset [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m). This repo consists of the code of CLIP-ViP model, the post-pretraining method, finetuning on text-to-video retrieval.


## Requirements 
We provide a Docker image for easier reproduction: `tiankaihang/azureml_docker:horovod`

We use mixed-precision training hence GPUs with Tensor Cores are recommended.


## Getting Started

### General

1. Download Data.

    Download HD-VILA-100M and other required data following the instruction of [HD-VILA](https://github.com/microsoft/XPretrain/tree/main/hd-vila). Also download auxiliary captions from: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/hdvila_ofa_captions_db.zip?sp=r&st=2023-03-16T04:58:26Z&se=2026-03-01T12:58:26Z&spr=https&sv=2021-12-02&sr=b&sig=EYE%2Bj11VWfQ6G5dZ8CKlOOpL3ckmmNqpAtUgBy3OGDM%3D)

2. Download pretrained models.

    We release the CLIP-ViP model under two settings:

    CLIP-ViP-B/32: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D)

    CLIP-ViP-B/16: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D)


### Pre-training

```bash
#inside the container
horovodrun -np $NUM_GPUS python src/pretrain/run_pretrain.py \
    --config $CONFIG_PATH
``` 

`$CONFIG_PATH` should be set to one of the .json config files available at [src/configs/pretrain](src/configs/pretrain). Currently, `pretrain_vip_base_32.json` and `pretrain_vip_base_16.json` are supported

### Text-to-Video Retrieval Finetuning

1. setting for accelerate (only at the first time)

```bash
accelerate config --config_file [path/to/store/config_file] 
```

2. wandb setting 

```bash
wandb login 
wandb online
```

3. running the code 

```bash
cd ./CLIP-ViP
bash src/tasks/run.sh
```

`$CONFIG_PATH` should be set to one of the .json config files available at [src/configs](src/configs) postfixed with `_retrieval.json`. For example, you can use `src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json` for finetuning CLIP-ViP-B/32 on MSRVTT retrieval. For model, currently, `pretrain_vip_base_32.json` and `pretrain_vip_base_16.json` are supported. For dataset, MSR-VTT, DiDemo, LSMDC, ActivityNet Captions are supported.

4. test the code 

```bash
accelerate launch --config_file /home/s2/dayelee/accel.yaml ./run_video_retrieval_val.py --config /home/s2/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/s2/dayelee --num_train_epochs 5 --output_dir /shared/s2/lab01/dayelee_store/checkpoints/msrvtt --train_batch_size 8 --e2e_weights_path /home/s2/dayelee/dayelee_store/checkpoints/msrvtt_16_gpu_2/epoch_5_bs_16_lr_1e-06_31/model_best.pt
```

make sure that you **do not** use `--is_train` and enter the `--e2e_weights_path` for loading a model checkpoint 

## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper:

```
@article{xue2022clip,
  title={CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment},
  author={Xue, Hongwei and Sun, Yuchong and Liu, Bei and Fu, Jianlong and Song, Ruihua and Li, Houqiang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2209.06430},
  year={2022}
}

@inproceedings{xue2022advancing,
  title={Advancing high-resolution video-language representation with large-scale video transcriptions},
  author={Xue, Hongwei and Hang, Tiankai and Zeng, Yanhong and Sun, Yuchong and Liu, Bei and Yang, Huan and Fu, Jianlong and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5036--5045},
  year={2022}
}
```

## Acknowledgements
The code is based on [HD-VILA](https://github.com/microsoft/XPretrain/tree/main/hd-vila).
