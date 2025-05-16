import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class CLModel(nn.Module):
    def __init__(self, args, n_classes, tokenizer=None):
        super().__init__()
        self.args = args
        # поддерживаем отсутствие dropout в args
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = n_classes
        # pad и mask тоже могут отсутствовать
        self.pad_value = getattr(args, 'pad_value', 1)
        self.mask_value = getattr(args, 'mask_value', 2)

        # базовый BERT
        self.f_context_encoder = AutoModel.from_pretrained(args.bert_path)
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        self.eps = 1e-8
        self.device = "cuda" if getattr(args, 'cuda', False) else "cpu"

        # классификатор
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )
        # функция отображения
        lower_dim = getattr(args, 'mapping_lower_dim', self.dim)
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, lower_dim),
        ).to(self.device)

        self.tokenizer = tokenizer

        # загружаем якори
        anchor_path = getattr(args, 'anchor_path', None)
        ds = getattr(args, 'dataset_name', None)
        if ds == "IEMOCAP":
            anchor_file = f"{anchor_path}/iemocap_emo.pt"
            self.emo_label = torch.tensor([0,1,2,3,4,5]).to(self.device)
        elif ds == "MELD":
            anchor_file = f"{anchor_path}/meld_emo.pt"
            self.emo_label = torch.tensor([0,1,2,3,4,5,6]).to(self.device)
        elif ds == "EmoryNLP":
            anchor_file = f"{anchor_path}/emorynlp_emo.pt"
            self.emo_label = torch.tensor([0,1,2,3,4,5,6]).to(self.device)
        else:
            anchor_file = None
            self.emo_label = None
        self.emo_anchor = torch.load(anchor_file).to(self.device) if anchor_file else None

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps

    def _encode_modality(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Закодировать вспомогательную модальность (визуал, аудио и т.п.)
        """
        mask = (ids != self.pad_value).long()
        cap = self.f_context_encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        emb = cap[:, 0, :]
        del cap
        return emb

    def _forward(self,
                 sentences: torch.Tensor,
                 vis_ids: torch.Tensor = None,
                 aud_ids: torch.Tensor = None,
                 bio_ids: torch.Tensor = None,
                 aus_ids: torch.Tensor = None):
        # attention mask для текста
        mask = (sentences != self.pad_value).long()
        # берем эмбеддинги токенов
        utterance_embs = self.f_context_encoder.embeddings(sentences.to(self.device))
        # подменяем CLS-позиции, если модальность есть
        if vis_ids is not None:
            vis_emb = self._encode_modality(vis_ids.to(self.device))
            utterance_embs[:, 1] = vis_emb
        if aud_ids is not None:
            aud_emb = self._encode_modality(aud_ids.to(self.device))
            utterance_embs[:, 2] = aud_emb
        if bio_ids is not None:
            bio_emb = self._encode_modality(bio_ids.to(self.device))
            utterance_embs[:, 3] = bio_emb
        if aus_ids is not None:
            aus_emb = self._encode_modality(aus_ids.to(self.device))
            utterance_embs[:, 4] = aus_emb

        # прогоняем через BERT
        utterance_encoded = self.f_context_encoder(
            inputs_embeds=utterance_embs,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']

        # находим position <mask>
        mask_pos = (sentences.to(self.device) == self.mask_value).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos]
        mask_mapped_outputs = self.map_function(mask_outputs)

        # классификатор
        feature = torch.dropout(mask_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)

        anchor_scores = None
        if getattr(self.args, 'use_nearest_neighbour', False) and self.emo_anchor is not None:
            anchors = self.map_function(self.emo_anchor)
            self.last_emo_anchor = anchors
            anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1),
                                           anchors.unsqueeze(0))
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores

    def forward(self,
                sentences: torch.Tensor,
                vis_ids: torch.Tensor = None,
                aud_ids: torch.Tensor = None,
                bio_ids: torch.Tensor = None,
                aus_ids: torch.Tensor = None,
                return_mask_output: bool = False):
        out = self._forward(sentences, vis_ids, aud_ids, bio_ids, aus_ids)
        if return_mask_output:
            return out
        return out[0]

class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + 1e-8

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / getattr(self.args, 'temp', 0.5)
