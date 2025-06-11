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
from matplotlib.figure import Figure

from model.model import CLModel

import streamlit as st

st.set_page_config(
    page_title="Emotion-anchor demo",
    layout="wide"
)
st.markdown("""
<style>
.icon-img {display:block; margin:0 auto;}
.icon-label{font-size:11px;text-align:center;margin-top:2px;}
.disabled-row {filter:grayscale(100%);opacity:.35;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
      div.stButton{
        display:flex;
        justify-content:flex-end;   
        padding-top:6px;
      }

      div.stButton>button{
        background:#E53935;
        color:#fff;
        padding:14px 56px;
        font-size:20px;
        border:none;
        border-radius:8px;
      }
      div.stButton>button:hover{
        background:#C62828;
      }
    </style>
    """,
            unsafe_allow_html=True
            )

CLASSES = ['neu', 'exc', 'fru', 'sad', 'hap', 'ang']
MODAL_NAMES = ['VIS', 'AUD', 'BIO', 'AUS']
ICON = {
    "TEXT": "icons/notes.png",
    "VIS": "icons/eye.png",
    "AUD": "icons/speaker.png",
    "BIO": "icons/user.png",
    "AUS": "icons/laugh.png",
}
FIXED_COLORS = {
    "neu": "#808080",
    "exc": "#32cd32",
    "fru": "#964B00",
    "sad": "#5b8ff9",
    "hap": "#FF8C00",
    "ang": "#DC143C",
}


def b64_icon(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


ICON64 = {k: b64_icon(v) for k, v in ICON.items()}  # ICON словарь был раньше

WEIGHTS_ROOT = Path("./weights")
EMOJI = {
    "neu": "😐",
    "exc": "😁",
    "fru": "😞",
    "sad": "😢",
    "hap": "😄",
    "ang": "😡",
}


def emoji_png(label: str, size_px: int = 120) -> str:
    emoji_char = EMOJI.get(label, "❓") + "\uFE0E"  # VS-15 = text glyph
    fig = Figure(figsize=(size_px / 80, size_px / 80), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, emoji_char,
            ha="center", va="center",
            fontsize=size_px * 0.8,
            color=FIXED_COLORS.get(label, "#888888"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"<img src='data:image/png;base64,{b64}' width='{size_px}' height='{size_px}'>"


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


# PATCH P – вставьте вместо старой tsne_figure
def tsne_figure(obj_vec: torch.Tensor,
                anchor_vecs: torch.Tensor,
                distances: list[float],
                pred_label: str) -> plt.Figure:
    data = torch.cat([obj_vec, anchor_vecs], dim=0).numpy()
    coords = TSNE(n_components=2, random_state=42,
                  perplexity=min(5, len(data) - 1)).fit_transform(data)
    obj_xy, anchors_xy = coords[0], coords[1:]

    dirs = anchors_xy - obj_xy
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit = dirs / (norms + 1e-8)

    max_dist = max(distances)
    radius_scale = 70 / max_dist
    full_lengths = np.array(distances) * radius_scale

    offset = 12.0

    new_xy = np.stack([
        obj_xy + u * max(l - offset, 0)
        for u, l in zip(unit, full_lengths)
    ], axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    for (end_x, end_y), lab in zip(new_xy, CLASSES):
        col = FIXED_COLORS.get(lab, "#888888")
        ax.plot([obj_xy[0], end_x], [obj_xy[1], end_y],
                color=col, linewidth=2, alpha=0.8)
    for (x, y), lab in zip(new_xy, CLASSES):
        col = FIXED_COLORS.get(lab, "#888888")
        ax.scatter(x, y + 1.5,
                   s=700,
                   facecolor='white',
                   edgecolor='none',
                   zorder=3)
        ax.text(x, y, EMOJI[lab],
                fontsize=32, ha='center', va='center',
                color=col, zorder=4)

    for dir_vec, l, lab in zip(unit, distances, CLASSES):
        midpoint = obj_xy + dir_vec * ((l * radius_scale - offset) / 2.0)
        col = FIXED_COLORS.get(lab, "#888888")
        ax.text(midpoint[0], midpoint[1], f"{l:.2f}",
                fontsize=8, color=col,
                ha='center', va='center', zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
                )

    ax.scatter(obj_xy[0], obj_xy[1],
               marker='x', s=280, c='black', zorder=6)

    circle = plt.Circle(obj_xy, radius_scale * max_dist,
                        linestyle='--', color='grey',
                        fill=False, alpha=0.4, linewidth=1.5)
    ax.add_patch(circle)

    ax.axis('off')
    fig.tight_layout()
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
left_top, right_top = st.columns([1.2, 1.8])
with left_top:
    col_ic, col_txt, col_ck = st.columns([0.11, 0.74, 0.15])
    with col_txt:
        preset = st.selectbox("Demo Preset", list(PRESETS.keys()))

left, right = st.columns([1.2, 1.8])
field_labels = {modal: name for modal, name in zip(["TEXT"] + MODAL_NAMES, ["TEXT", "VISUAL", "AUDIO", "BIO", "AUS"])}

with left:
    default_text, default_vis, default_aud, default_bio, default_aus, gt_label = PRESETS[preset]

    modal_inputs = []
    default_vals = [default_vis, default_aud, default_bio, default_aus]
    rows = [("TEXT", default_text)] + list(zip(MODAL_NAMES, default_vals))

    for label, default_val in rows:
        col_ic, col_txt, col_ck = st.columns([0.11, 0.74, 0.15])

        if label == "TEXT":
            active = True
        else:
            with col_ck:
                active = st.checkbox(
                    "", value=bool(default_val), key=f"{label}_active",
                    label_visibility="collapsed"
                )

        row_cls = "" if active else "disabled-row"

        with col_ic:
            st.markdown(
                f"""
                <div class="{row_cls}">
                  <br /><br />
                  <img class="icon-img" src="data:image/png;base64,{ICON64[label]}" width="40">
                  <div class="icon-label" style='font-size:20px;'>{field_labels[label]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_txt:
            entry = st.text_area(
                "", value=default_val if label != "TEXT" else default_text,
                key=f"{label}_txt", height=90,
                placeholder=f"Enter {label.lower()} here...",
                disabled=not active
            )
            if label == "TEXT":
                text = entry  # обновляем utterance
            else:
                modal_inputs.append(entry if active and entry.strip() else None)

        if not active and label != "TEXT":
            st.markdown(
                f"<style>#{col_ck._form_script_id} input{{filter:grayscale(100%);opacity:.35;}}</style>",
                unsafe_allow_html=True
            )

    col_ic, col_txt, col_ck = st.columns([0.11, 0.74, 0.15])
    with col_txt:
        run_clicked = st.button("RUN")

if run_clicked:
    if not text.strip():
        left.warning("Please enter the utterance text.")
        st.stop()

    with st.spinner("Calculating…"):
        distances, obj_vec, anchor_vecs = compute_anchor_distances(
            [text] + modal_inputs, args
        )

    best_idx = int(np.argmin(distances))
    pred_label = CLASSES[best_idx]

    fig_tsne = tsne_figure(obj_vec, anchor_vecs, distances, pred_label)

    name_map = {
        "neu": "Neutral",
        "exc": "Excited",
        "fru": "Frustrated",
        "sad": "Sad",
        "hap": "Happy",
        "ang": "Angry",
    }

    gt_name = name_map.get(gt_label, "-")
    pred_name = name_map[pred_label]

    gt_html = emoji_png(gt_label) if gt_label else "-"
    pred_html = emoji_png(pred_label)

    right.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)

    graph_col, info_col = right.columns([1.2, 0.6])

    info_col.markdown("<div style='height:200px'></div>", unsafe_allow_html=True)

    graph_col.pyplot(fig_tsne)

    with info_col:
        name = {
            "neu": "Neutral", "exc": "Excited", "fru": "Frustrated",
            "sad": "Sad", "hap": "Happy", "ang": "Angry",
        }

        gt_name = name.get(gt_label, "-")
        pred_name = name[pred_label]

        gt_html = emoji_png(gt_label) if gt_label else "-"
        pred_html = emoji_png(pred_label)

        info_col.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:24px;
                        font-size:24px;line-height:24px;">

              <div style="display:flex;align-items:center;gap:14px;">
                {gt_html}
                <div>
                  <b>Ground&nbsp;Truth</b><br>{gt_name}
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:14px;">
                {pred_html}
                <div>
                  <b>Prediction</b><br>{pred_name}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    with right_top:
        st.info("Submit data and press Run.")