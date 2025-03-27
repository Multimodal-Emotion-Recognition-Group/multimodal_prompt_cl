from config import *
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class HybridLossOutput:
    ce_loss: torch.Tensor = None
    cl_loss: torch.Tensor = None
    sentiment_representations: torch.Tensor = None
    sentiment_labels: torch.Tensor = None
    sentiment_anchortypes: torch.Tensor = None
    anchortype_labels: torch.Tensor = None
    max_cosine: torch.Tensor = None

@dataclass
class TripletOutput:
    loss: torch.Tensor = None
    sentiment_representations: torch.Tensor = None
    sentiment_labels: torch.Tensor = None
    sentiment_anchortypes: torch.Tensor = None
    anchortype_labels: torch.Tensor = None
    max_cosine: torch.Tensor = None

class TripletLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.margin = args.triplet_margin if hasattr(args, 'triplet_margin') else 0.2
        if args.dataset_name == "IEMOCAP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5])
        elif args.dataset_name == "IEMOCAP4":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap4_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3])
        elif args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "EmoryNLP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/emorynlp_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        self.args = args
    
    def forward(self, reps, labels, model, return_representations=False):
        device = reps.device
        batch_size = reps.shape[0]
        
        if not self.args.disable_emo_anchor:
            self.emo_anchor = self.emo_anchor.to(device)
            self.emo_label = self.emo_label.to(device)
            emo_anchor_mapped = model.map_function(self.emo_anchor) 
            
            positive_anchor = emo_anchor_mapped[labels]  
            cos_sim = F.cosine_similarity(reps, positive_anchor, dim=-1)
            d_pos = 1 - cos_sim  
            
            reps_expanded = reps.unsqueeze(1)                   
            anchors_expanded = emo_anchor_mapped.unsqueeze(0)
            cos_sim_all = F.cosine_similarity(reps_expanded, anchors_expanded, dim=-1)  
            d_all = 1 - cos_sim_all  
            
            labels_expanded = labels.unsqueeze(1)               
            emo_label_expanded = self.emo_label.unsqueeze(0)      
            neg_mask = (labels_expanded != emo_label_expanded)       
            d_neg = torch.where(neg_mask, d_all, torch.tensor(float('inf')).to(device))
            d_neg, _ = d_neg.min(dim=1)  # (batch_size,)
            
            loss = F.relu(d_pos - d_neg + self.margin).mean()
            
            if return_representations:
                sentiment_labels = labels
                sentiment_representations = reps.detach()
                sentiment_anchortypes = emo_anchor_mapped.detach()
            else:
                sentiment_labels = None
                sentiment_representations = None
                sentiment_anchortypes = None
                
            norm_anchors = F.normalize(emo_anchor_mapped, p=2, dim=1)
            cosine_matrix = torch.matmul(norm_anchors, norm_anchors.t())
            cosine_matrix = cosine_matrix - 2 * torch.diag(torch.diag(cosine_matrix))
            max_cosine = cosine_matrix.max().clamp(-0.99999, 0.99999)
        
        else:
            norm_reps = F.normalize(reps, p=2, dim=1)
            cosine_matrix = torch.matmul(norm_reps, norm_reps.t())
            dists = 1 - cosine_matrix  # (batch_size, batch_size)
            labels_expanded = labels.unsqueeze(0)
            mask = (labels_expanded == labels_expanded.t())
            eye = torch.eye(batch_size, device=device).bool()
            mask_no_self = mask & ~eye
            
            if mask_no_self.sum() > 0:
                d_pos, _ = dists.masked_fill(~mask_no_self, -1e9).max(dim=1)
            else:
                d_pos = torch.zeros(batch_size, device=device)
            mask_neg = ~mask
            d_neg, _ = dists.masked_fill(~mask_neg, 1e9).min(dim=1)
            loss = F.relu(d_pos - d_neg + self.margin).mean()
            
            if return_representations:
                sentiment_labels = labels
                sentiment_representations = reps.detach()
                sentiment_anchortypes = None
            else:
                sentiment_labels = None
                sentiment_representations = None
                sentiment_anchortypes = None
            max_cosine = None
        
        return TripletOutput(
            loss=loss,
            sentiment_representations=sentiment_representations,
            sentiment_labels=sentiment_labels,
            sentiment_anchortypes=sentiment_anchortypes,
            anchortype_labels=self.emo_label if not self.args.disable_emo_anchor else None,
            max_cosine=max_cosine
        )

def loss_function(log_prob, reps, label, mask, model, args):
    if hasattr(model, 'module'):
        model = model.module

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(reps.device)
    triplet_loss_fn = TripletLoss(args)
    triplet_loss_output = triplet_loss_fn(reps, label, model, return_representations=not model.training)
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=triplet_loss_output.loss,
        sentiment_representations=triplet_loss_output.sentiment_representations,
        sentiment_labels=triplet_loss_output.sentiment_labels,
        sentiment_anchortypes=triplet_loss_output.sentiment_anchortypes,
        anchortype_labels=triplet_loss_output.anchortype_labels,
        max_cosine=triplet_loss_output.max_cosine
    )
