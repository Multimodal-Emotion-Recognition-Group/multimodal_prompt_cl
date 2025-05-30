import argparse
import random
from pathlib import Path
import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import cm

from model.model import CLModel

import streamlit as st
st.set_page_config(
    page_title="Emotion-anchor demo",
    layout="wide"
)

CLASSES     = ['neu', 'exc', 'fru', 'sad', 'hap', 'ang']
MODAL_NAMES = ['VIS', 'AUD', 'BIO', 'AUS']
WEIGHTS_ROOT = Path("./weights")                    
EMOJI = {
    "neu": "😐",
    "exc": "🤩",
    "fru": "😠",
    "sad": "😢",
    "hap": "😄",
    "ang": "😡",
}

def load_model(weights_file: Path, _args: argparse.Namespace):
    model = CLModel(args, n_classes=len(CLASSES), tokenizer=args.tokenizer).to(args.device)
    model.load_state_dict(torch.load(weights_file, map_location=args.device))
    model.eval()
    return model
    
def seed_everything(seed: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(2)


def select_weights_file(active_flags: list[bool]) -> str:
    return ('+'.join([n for f, n in zip(active_flags, MODAL_NAMES) if f]) or 'EACL') + '.pkl'


def compute_anchor_distances(values, args):
    active_flags = [v is not None for v in values[1:]]
    weights_file = WEIGHTS_ROOT / select_weights_file(active_flags)
    model = load_model(weights_file, args)

    text, vis_cap, aud_cap, bio, aus = values
    prompt = "For utterance: " + (text or "") + "  feels <mask> "
    token_ids = args.tokenizer(prompt)['input_ids'][1:]
    seq = token_ids[-args.max_len:]
    pad_len = args.max_len - len(seq)
    seq = seq + [args.pad_value] * pad_len
    p2 = torch.LongTensor(seq).unsqueeze(0).to(args.device)

    def make_modal(t: str):
        ids = args.tokenizer(t or "")['input_ids']
        ids = ids[-args.max_len:]
        pad = args.max_len - len(ids)
        return torch.LongTensor(ids + [args.pad_value] * pad).unsqueeze(0).to(args.device)

    vis = make_modal(vis_cap) if active_flags[0] else None
    aud = make_modal(aud_cap) if active_flags[1] else None
    bio = make_modal(bio) if active_flags[2] else None
    aus = make_modal(aus) if active_flags[3] else None

    with torch.no_grad():
        _, mask_mapped_output, _, anchor_scores = model(
            p2, vis, aud, bio, aus, return_mask_output=True
        )

        anchors = model.map_function(model.emo_anchor)             
        cosine_sims = F.cosine_similarity(
            mask_mapped_output.unsqueeze(1),   
            anchors.unsqueeze(0),              
            dim=-1
        ).squeeze(0)                           
        cosine_dists = 1.0 - cosine_sims       

    return cosine_dists.tolist(), mask_mapped_output.cpu(), anchors.cpu()


def tsne_figure(obj_vec: torch.Tensor,
                anchor_vecs: torch.Tensor,
                distances: list[float]) -> plt.Figure:
    data = torch.cat([obj_vec, anchor_vecs], dim=0).numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(data)-1))
    coords = tsne.fit_transform(data)
    obj_xy, anchors_xy = coords[0], coords[1:]

    dirs = anchors_xy - obj_xy
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_unit = np.where(norms == 0, 0, dirs / norms)
    max_dist = max(distances)
    radius_scale = 70 / max_dist            # 70 px для макс. дистанции
    new_anchors_xy = obj_xy + dirs_unit * (np.array(distances)[:, None] * radius_scale)

    norm = plt.Normalize(min(distances), max(distances))
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(norm(distances))

    fig, ax = plt.subplots()
    for (x, y), d, color in zip(new_anchors_xy, distances, colors):
        ax.scatter(x, y, s=80, color=color, edgecolor='k')
    for i, (x, y) in enumerate(new_anchors_xy):
        ax.text(x, y-4, CLASSES[i], fontsize=9, ha='center', va='top', fontweight='bold')

    ax.scatter(obj_xy[0], obj_xy[1], marker='x', s=140, c='black', label='Object')
    circ = plt.Circle(obj_xy, radius_scale * max_dist, color='grey',
                      linestyle='--', fill=False, alpha=0.3)
    ax.add_patch(circ)
    ax.axis('off')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_ticks([min(distances), max(distances)])
    cbar.set_ticklabels([f'{min(distances):.2f}', f'{max(distances):.2f}'])
    cbar.set_label('Cosine distance')

    return fig


# Args

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', type=str, default='princeton-nlp/sup-simcse-roberta-base')
parser.add_argument('--anchor_path', type=str, default='./emo_anchors/sup-simcse-roberta-base')
parser.add_argument('--use_nearest_neighbour', action='store_true', )
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--dataset_name', type=str, default='IEMOCAP')

parser.add_argument('--wp', type=int, default=8)
parser.add_argument('--wf', type=int, default=0)
parser.add_argument('--pad_value', type=int, default=1)
parser.add_argument('--mask_value', type=int, default=2)
parser.add_argument('--max_len', type=int, default=256)

parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--mapping_lower_dim', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--seed', type=int, default=2)

args, _ = parser.parse_known_args()

args.device = torch.device('cpu')
args.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
args.tokenizer.add_tokens(["<mask>", "[VIS]", "[AUD]", "[BIO]", "[AUS]"])

# Demo presets

PRESETS = {
    "Custom": ("", "", "", "", "", None),
    "MELD sample": (
        "Why do all you're coffee mugs have numbers on the bottom?",
        "The image depicts a scene from a sitcom set in a kitchen, likely from the popular series 'Friends.' The setting is vibrant with a mix of warm and cool tones. The kitchen features blue cabinets, a white refrigerator adorned with magnets and notes, and various kitchen items like bottles, jars, and utensils scattered across the countertops. The walls are painted a soft yellow, adding to the cozy ambiance. The characters, a woman in a light-colored dress and a man in a dark suit, stand near the counter, engaged in conversation. The woman holds a phone, suggesting she might be showing something to the man or discussing it. The overall mood appears casual and friendly, typical of a sitcom setting where everyday life is portrayed humorously. The composition centers on the interaction between the two characters, emphasizing their expressions and body language within the familiar domestic environment.",
        "The speaker's voice has a distinctive raspy quality with a slightly deep pitch. It moves at a moderate tempo, indicating neither rush nor slowness but rather a steady, relaxed pace. The intonation carries a subtle up-and-down movement which adds depth to the speech, making it more expressive and nuanced. There is an underlying tone of happiness, reflecting a positive emotional state. Additionally, there is a noticeable lisp in his speech, giving it a unique texture and rhythm.",
        "Mark appears curious and observant, likely paying attention to small details in his environment. His questioning suggests a keen interest in understanding the reasoning behind objects or practices around him. This curiosity indicates he values knowledge and possibly enjoys learning new things from his friends.",
        "Inner Brow Raiser, Brow Lowerer, Lip Corner Depressor",
        "exc"
    ),
    "IEMOCAP sample": (
        "Sure. Everybody is told the same thing, it keeps us all excited. It keeps us all coming back for more. It keeps us thinking that life is gonna start now any minute if we can just find the right spot and get in on the action.",
        "The image appears to be a split-screen video capture showing two individuals engaged in what seems to be an interview or discussion setting. The left side shows a person seated against a plain white wall, wearing a dark long-sleeved shirt and light-colored pants. They are gesturing with their right hand, suggesting they are speaking or explaining something. The right side features another individual, also seated, facing slightly away from the camera, wearing a dark top and what looks like a head covering. This person's posture suggests they might be listening attentively. In the background of the right frame, there is a glimpse of another individual seated further back, possibly indicating a panel or audience setting. The room has a simple, functional appearance with neutral colors and minimal decoration, which could suggest a professional or educational environment. The overall mood appears to be focused and conversational.",
        "The speaker's voice has a light and airy quality with a slightly high pitch. It moves quickly over the words indicating a lively and spirited manner. The intonation is consistent, creating a sense of stability and confidence. There's an underlying happiness in the speaker's voice which makes it sound vibrant and engaging.",
        "Speaker F appears to be a thoughtful and reflective individual, often pausing before responding, which suggests careful consideration. They contribute insightful comments, indicating a deeper understanding or interest in the topic. However, their responses are somewhat sporadic, suggesting they might not be as verbally assertive or dominant in the conversation.",
        "Cheek Raiser, Lip Corner Puller",
        "fru"
    ),
}

# UI

st.title("Emotion-anchor demo")
left, right = st.columns([1.9, 1.1])    

with left:
    preset = st.selectbox("Demo Preset", list(PRESETS.keys()))
    default_text, default_vis, default_aud, default_bio, default_aus, gt_label = PRESETS[preset]

    text = st.text_area(
        "Utterance text:",
        value=default_text,
        height=100,
        placeholder="Type your utterance here..."
    )

    modal_inputs = []
    default_vals = [default_vis, default_aud, default_bio, default_aus]
    for name, default_val in zip(MODAL_NAMES, default_vals):
        active = st.checkbox(name, value=bool(default_val), key=f"{name}_active")
        val = st.text_area(                         # text_area → перенос строк, помещается весь текст
            f"{name} value",
            value=default_val,
            height=70,
            key=name,
            placeholder=f"Enter {name.lower()}..."
        )
        modal_inputs.append(val if active and val.strip() else None)

    run_clicked = st.button("Run")
    
if run_clicked:
    if not text.strip():
        left.warning("Please enter the utterance text.")   # выводим в той же колонке
        st.stop()

    with st.spinner("Calculating…"):
        distances, obj_vec, anchor_vecs = compute_anchor_distances(
            [text] + modal_inputs, args
        )

    fig_tsne = tsne_figure(obj_vec, anchor_vecs, distances)
    
    best_idx = int(np.argmin(distances))
    pred_label  = CLASSES[best_idx]
    pred_emoji  = EMOJI[pred_label]
    gt_emoji    = EMOJI.get(gt_label, "❓") if gt_label else "—"

    right.markdown(
        f"<div style='text-align:right; font-size:56px;'>"
        f"GROUND&nbsp;TRUTH {gt_emoji}<br>"
        f"PREDICTION&nbsp;{pred_emoji}"
        f"</div>",
        unsafe_allow_html=True
    )

    right.pyplot(fig_tsne)                
else:
    right.info("Submit data and press **Run**.")           # сообщение тоже справа
