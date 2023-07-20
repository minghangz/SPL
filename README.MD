# SPL: Generating Structured Pseudo Labels for Noise-resistant Zero-shot Video Sentence Localization

In this work, we propose the Structurebased Pseudo Label generation (SPL) framework for the zero-shot video sentence localization task, which learns with only video data without any annotation. 
We first generate free-form interpretable pseudo queries before constructing query-dependent event proposals by modeling the event temporal structure. To mitigate the effect of pseudolabel noise, we propose a noise-resistant iterative method that repeatedly re-weight the training sample based on noise estimation to train a grounding model and correct pseudo labels.
Experiments on the ActivityNet Captions and Charades-STA datasets demonstrate the advantages of our approach.

Our paper was accepted by ACL-2023. [[Paper]](https://aclanthology.org/2023.acl-long.794/) [[Project]](https://minghangz.github.io/publication/spl/)

![pipeline](imgs/pipeline.png)

## Main Results

![pipeline](imgs/results.png)

## Generate Pseudo Labels (optional)

We provide the generated pseudo labels for the ActivityNet Captions and Charades-STA datasets in `EMB/data/dataset/activitynet/train_pseudo.json` and `EMB/data/dataset/charades/charades_sta_train_pseudo.txt`. If you need to generate pseudo labels by yourself, please follow the instructions.

We use the [BLIP](https://github.com/salesforce/BLIP) model to generate pseudo labels. We provide pre-extracted BLIP captions and features at [this link](https://pan.baidu.com/s/1ys2WX78r6ym1wsFj4JRuXg?pwd=59yu). If you download the features and captions we provide, please proceed directly to step 3.

### Step 0. Environment

Please clone [BLIP](https://github.com/salesforce/BLIP) and copy the following files into the directory for BLIP:

```bash
git clone git@github.com:salesforce/BLIP.git
cp captions_generator.py feature_extractor.py BLIP
cd BLIP
```

Please follow [BLIP](https://github.com/salesforce/BLIP)'s instruction to prepare the python enviroment.

### Step 1. Generate Captions

You can download our generated captions from [here](https://pan.baidu.com/s/1ys2WX78r6ym1wsFj4JRuXg?pwd=59yu). You can also generate captions using the following script.

To generate captions, please run:

```bash
# Charades-STA
python captions_generator.py --video_root PATH_TO_VIDEO --save_root PATH_TO_SAVE_CAPTIONS

# ActivityNet Captions
python captions_generator.py --video_root PATH_TO_VIDEO --save_root PATH_TO_SAVE_CAPTIONS --stride 16 --num_stnc 5
```

### Step 2. Extract Features

You can download our extracted features from [here](https://pan.baidu.com/s/1ys2WX78r6ym1wsFj4JRuXg?pwd=59yu). You can also extract the features using the following scripts.

To extract the visual features, please run:

```bash
# Charades-STA
python feature_extractor.py --input_root PATH_TO_VIDEO --save_root PATH_TO_SAVE_VISUAL_FEATURES

# ActivityNet Captions
python feature_extractor.py --input_root PATH_TO_VIDEO --save_root PATH_TO_SAVE_VISUAL_FEATURES --stride 16
```

To extract the caption features, please run:

```bash
python feature_extractor.py --input_root PATH_TO_SAVED_CAPTIONS --save_root PATH_TO_SAVE_CAPTION_FEATURES --extract_text
```

### Step 3. Generate Pseudo Labels

To generate the pseudo labels, please run:

```bash
# leave BLIP's directory
cd ..

# Charades-STA
python pseudo_label_generation.py --dataset charades --video_feat_path PATH_TO_SAVED_VISUAL_FEATURES --caption_feat_path PATH_TO_SAVED_CAPTION_FEATURES --caption_path PATH_TO_SAVED_CAPTIONS

# ActivityNet Captions
python pseudo_label_generation.py --dataset activitynet --num_stnc 4 --stnc_th 0.9 --stnc_topk 1 --video_feat_path PATH_TO_SAVED_VISUAL_FEATURES --caption_feat_path PATH_TO_SAVED_CAPTION_FEATURES --caption_path PATH_TO_SAVED_CAPTIONS
```

Note: On the ActivityNet Captions dataset, due to the low efficiency of the sliding window method for generating event proposals, we pre-reduced the number of event proposals by clustering features. The processed proposals are stored in `data/activitynet/events.pkl`. The preprocessing script is `event_preprocess.py`.

## Quick Start

### Environment

We use [EMB](https://github.com/Raymond-sci/EMB) as our grounding model and train it using our generated pseudo labels. Please follow [EMB](https://github.com/Raymond-sci/EMB)'s instruction to download features and prepare python envrionment.

### Training

To train EMB with generated pseudo labels, please run:
```bash
cd EMB

# Charades-STA
python main.py --task charades --mode train --deploy

# ActivityNet Captions
python main.py --task activitynet --mode train --deploy
```

### Evaluation

Download our trained models from [here](https://drive.google.com/drive/folders/1ET_izmDizcu-wUtCHVzk3n1r-B_P3hph?usp=sharing) and put them in `EMB/sessions/`. Create the `session` folder if not existed.

To evaluate the trained model, please run:

```bash
cd EMB

# Charades-STA
python main.py --task charades --mode test --model_name SPL

# ActivityNet Captions
python main.py --task activitynet --mode test --model_name SPL
```

## Acknowledege

We appreciate [EMB](https://github.com/Raymond-sci/EMB) for its implementation. We use EMB as our grounding model.

## Citation

```
@inproceedings{zheng-etal-2023-generating,
    title = "Generating Structured Pseudo Labels for Noise-resistant Zero-shot Video Sentence Localization",
    author = "Zheng, Minghang  and
      Gong, Shaogang  and
      Jin, Hailin  and
      Peng, Yuxin  and
      Liu, Yang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.794",
    pages = "14197--14209",
}
```
