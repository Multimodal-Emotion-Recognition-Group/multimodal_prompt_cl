# Multimodal Prompt + EACL

- env.yml - conda environment file
- run.sh - old EACL script
- valid set is empty -> val loss and metrics is nan

## Launch

1. `python src/generate_anchors.py --bert_path <model_path>` - generate emotion anchors (there are already generated anchors for RoBERTa-large)
2. `python src/run.py --anchor_path "./emo_anchors/sup-simcse-roberta-large" --use_nearest_neighbour`

## Experiments

- `--max_len` set 512 (default value is 256)
- Try to train with batch_size=64 in fp32
- Try different metrics for stage 2 (now cosine_similarity)
- Think about another algorithm of Stage 2 prediction

## Branch Navigation

| Branch | Purpose |
|--------|---------|
| main | Stable, production-ready code |
| demo | Streamlit web demo for the ACM Multimedia conference |
| parallel | Variant of main adapted for DDP training |
| exp_multimodal_desc | Experimental multimodal-description implementation |
| exp_triplet_loss | Experimental Triplet-Loss implementation |
