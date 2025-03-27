# Multimodal Prompt + EACL

- env.yml - conda environment file
- run.sh - old EACL script
- valid set is empty -> val loss and metrics is nan

## Launch
1. `python src/generate_anchors.py --bert_path <model_path>` - generate emotion anchors (there are already generated anchors for RoBERTa-large)
2. `torchrun --nproc_per_node=N_GPU ./src/run.py --epochs 10 --batch_size 16 --dataset_name MELD --anchor_path ./emo_anchors/sup-simcse-roberta-large --use_nearest_neighbour`

## Fixed anchors setup
1. `python src/generate_fixed_anchors.py`

    - --dim - embedding size (1024 - Roberta-large, 768 - Roberta-base)
    
    - --save_path (expects `./emo_anchors/fixed_anchors` for Roberta-large and `./emo_anchors/fixed_anchors_768` for Roberta-base)

2. `torchrun --nproc_per_node=N_GPU ./src/run.py --epochs 10 --batch_size 16 --dataset_name MELD --anchor_path ./emo_anchors/fixed_anchors --use_nearest_neighbour --use_pretrained`

    - --bert_path - `princeton-nlp/sup-simcse-roberta-large` or `princeton-nlp/sup-simcse-roberta-large`

    - --use_pretrained - if set load weights from HuggingFace otherwise load only model config and build model with random weights

    - --dataset_name - MELD, IEMOCAP, IEMOCAP4