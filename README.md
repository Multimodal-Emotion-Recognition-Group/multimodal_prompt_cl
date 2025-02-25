# Multimodal Prompt + EACL

- env.yml - conda environment file
- run.sh - old EACL script
- valid set is empty -> val loss and metrics is nan

## Launch
1. `python src/generate_anchors.py --bert_path <model_path>` - generate emotion anchors (there are already generated anchors for RoBERTa-large)
2. `torchrun --nproc_per_node=N_GPU ./src/run.py --epochs 10 --batch_size 16 --dataset_name MELD --anchor_path ./emo_anchors/sup-simcse-roberta-large --use_nearest_neighbour`

## Experiments
- `--max_len` set 512 (default value is 256)
- Try to train with batch_size=64 in fp32
- Try different metrics for stage 2 (now cosine_similarity)
- Think about another algorithm of Stage 2 prediction