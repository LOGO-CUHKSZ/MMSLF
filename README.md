# Improving Task-Specific Multimodal Sentiment Analysis with General MLLMs via Prompting

Pytorch implementation of the paper: 
> **[Improving Task-Specific Multimodal Sentiment
Analysis with General MLLMs via Prompting](https://openreview.net/pdf?id=PBy1Ew1ihV)**

> This is a reorganized code, if you find any bugs please contact me. Thanks.

## Content
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training Instructions](#Training-instructions)
- [Citation](#Citation)

## Data Preparation
1. Use `preprocess/Generate_Prompt.py` to generate prompts. You need to set the base_url, API Key, and model (e.g., GPT-4o-omni). After running, it will output a csv file.
2. Use `preprocess/Construct_Data_Prompt.py` to combine the prompts saved in the csv with the dataset files to complete data preprocessing.

## Environment
The basic training environment for the results in the paper is Pytorch 2.1.1 with NVIDIA Tesla A40 (CUDA 12.1). 

## Training Instructions
Take the SIMS dataset as an example:
```
python train_teacher.py --project_name sims_gpt_teacher_05 --CUDA_VISIBLE_DEVICES '0' --datasetName 'sims' --dataPath './SIMS/unaligned_39_prompt.pkl' --bert_finetune --batch_size 64 --n_epochs 100 --lr 0.0002 --min_sampling_rate 0.5 --seed 1111

python train_student.py --project_name stu_sims_1111 --teacher_project_name sims_gpt_teacher_05 --CUDA_VISIBLE_DEVICES '0' --datasetName 'sims' --dataPath './SIMS/unaligned_39_prompt.pkl' --encoder_depth 1 --fusion_depth 2 --batch_size 64 --n_epochs 200 --lr 0.0001 --alpha 1.0 --beta 60.0 --gamma 8.0 --seed 1111
```

## Citation

- [Improving Task-Specific Multimodal Sentiment Analysis with General MLLMs via Prompting](https://openreview.net/pdf?id=PBy1Ew1ihV)

Please cite our paper if you find our work useful for your research:
```
@inproceedings{zhang-etal-2025-mmslf,
    title = "Improving Task-Specific Multimodal Sentiment Analysis with General MLLMs via Prompting",
    author = "Zhang, Haoyu and 
              Zhang, Yinan and 
              Ying, Chaolong and 
              Tang, Xiaoying and 
              Yu, Tianshu",
    booktitle = "The Thirty-Nine Annual Conference on Neural Information Processing Systems (NeurIPS 2025)",
    year = "2025"
}
```
