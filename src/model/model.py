import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

class CLModel(nn.Module):
    def __init__(self, args, n_classes, tokenizer=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = n_classes
        self.pad_value = args.pad_value
        self.mask_value = 50264
        # self.f_context_encoder = AutoModel.from_pretrained(args.bert_path)
        conf = AutoConfig.from_pretrained(args.bert_path)
        self.f_context_encoder = AutoModel.from_config(conf) 
        
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.avg_dist = []

        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        self.eps = 1e-8
        self.device = "cuda" if self.args.cuda else "cpu"
        self.predictor = nn.Sequential(
            # nn.Linear(self.dim, self.dim),
            # nn.ReLU(),
            nn.Linear(self.dim, self.num_classes)
        )
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)

        self.tokenizer = tokenizer

        if args.dataset_name == "IEMOCAP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5]).to(self.device)
        elif args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "IEMOCAP4":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3]).to(self.device)
        # elif args.dataset_name == "EmoryNLP":
        #     self.emo_anchor = torch.load(f"{args.anchor_path}/emorynlp_emo.pt").to(self.device)
        #     self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6]).to(self.device)

    def device(self):
        return self.f_context_encoder.device
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def _forward(self, sentences, vis_ids, aud_ids, bio_ids, aus_ids):
        mask = 1 - (sentences == (self.pad_value)).long()
        
        vis_ids = vis_ids.to(self.device)
        vis_cap = self.f_context_encoder(
            input_ids=vis_ids,
            attention_mask=torch.ones_like(vis_ids).long(),
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        vis_emb = vis_cap[:, 0, :]
        del vis_cap

        aud_ids = aud_ids.to(self.device)
        aud_cap = self.f_context_encoder(
            input_ids=aud_ids,
            attention_mask=torch.ones_like(aud_ids).long(),
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        aud_emb = aud_cap[:, 0, :]
        del aud_cap

        bio_ids = bio_ids.to(self.device)
        bio_cap = self.f_context_encoder(
            input_ids=bio_ids,
            attention_mask=torch.ones_like(bio_ids).long(),
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        bio_emb = bio_cap[:, 0, :]
        del bio_cap

        aus_ids = aus_ids.to(self.device)
        aus_cap = self.f_context_encoder(
            input_ids=aus_ids,
            attention_mask=torch.ones_like(aus_ids).long(),
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        aus_emb = aus_cap[:, 0, :]
        del aus_cap

        sentences = sentences.to(self.device)
        mask = mask.to(self.device)
        utterance_embs = self.f_context_encoder.embeddings(sentences)
        utterance_embs[:, 1] = vis_emb
        utterance_embs[:, 1] = aud_emb
        utterance_embs[:, 2] = bio_emb
        utterance_embs[:, 3] = aus_emb

        utterance_encoded = self.f_context_encoder(
            # input_ids=sentences,
            inputs_embeds=utterance_embs,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']

        

        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos]
        mask_mapped_outputs = self.map_function(mask_outputs)
        feature = torch.dropout(mask_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)
        if self.args.use_nearest_neighbour:
            anchors = self.map_function(self.emo_anchor)

            self.last_emo_anchor = anchors
            anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), anchors.unsqueeze(0))
            
        else:
            anchor_scores = None
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    
    def forward(self, sentences, vis_ids, aud_ids, bio_ids, aus_ids, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(sentences, vis_ids, aud_ids, bio_ids, aus_ids)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        else:
            return feature
        
class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
    
    def forward(self, emb):
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp
