# Multimodal Adaptive Emotion Anchoring

Identifying the emotional state of each statement in a conversation, known as Emotion Recognition in Conversations (ERC), is essential for developing empathetic systems. The field of ERC has witnessed substantial progress with the development of context-sensitive prompting techniques, such as Emotion-Anchored Contrastive Learning (EACL). The main problem is that existing approaches rely mainly on textual information without considering audio and visual cues. To address this limitation, we propose Multimodal Adaptive Emotion Anchoring (MAEA), an extension of EACL that incorporates multimodal signals and a dynamic anchor separation strategy. Our approach enhances emotion representation learning by integrating visual and acoustic features while refining anchor separation based on conversational contexts. By dynamically adapting anchor positions depending on conversational shifts and emotional transitions, MAEA enables more robust emotion modeling. Our experiments demonstrate that MAEA can improve unimodal setups and produce superior results on benchmark ERC datasets.

Preview video [link](https://drive.google.com/file/d/1hRlLbL4EiVOZY26aoFEymrF4vf9tR3Tv/view?usp=drive_link)

Demo video
[link](demo_video.mp4)

![pipeline2](https://github.com/user-attachments/assets/ad7e8151-1215-4d8d-9378-21fe4e5f3b73)

# Results

|Model|MELD w-F1|IEMOCAP w-F1|
| :------ | ----: | ----: |
|UniMSE|65.51|70.66|
|M2FNet|66.81|69.69|
|SACL-LSTM|66.86|69.22|
|HiDialog|66.96|-|
|DF-ERC|67.03|71.75|
|EACL|67.12|70.41|
|SPCL-CL-ERC|65.51|70.66|
|TelMe|67.37|70.48|
|Mamba-like Model|67.60|-|
|SpeechCueLLM|67.60|**72.60**|
|**Ours**|67.27|70.25|
|**Ours w/correction**|**67.69**|70.53|

## Launch
1. `python src/generate_anchors.py --bert_path <model_path>` - generate emotion anchors (there are already generated anchors for RoBERTa-large)
2. `python src/run.py --anchor_path "./emo_anchors/sup-simcse-roberta-large" --use_nearest_neighbour`
