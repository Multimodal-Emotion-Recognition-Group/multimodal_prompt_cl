import io
import base64
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from flask import Flask, request, render_template

import torch

from model.model import CLModel

matplotlib.use('Agg')

app = Flask(__name__)
CLASSES = ['neu', 'exc', 'fru', 'sad', 'hap', 'ang']
MODAL_NAMES = ['VIS', 'AUD', 'BIO', 'AUS']

WEIGHTS_ROOT = Path("../weights")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_weights_file(active_flags: list[bool]) -> str:
    if not any(active_flags):
        return 'EACL.pkl'
    names = [name for flag, name in zip(active_flags, MODAL_NAMES) if flag]
    return '+'.join(names) + '.pkl'


def compute_anchor_distances(values, args):
    active_flags = [v is not None for v in values[1:]]
    model = CLModel(args, n_classes=len(CLASSES), tokenizer=args.tokenizer).to(args.device)
    weights_file = WEIGHTS_ROOT / select_weights_file(active_flags)
    model.load_state_dict(torch.load(weights_file, map_location=args.device))
    model.eval()

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
            p2,
            vis,
            aud,
            bio,
            aus,
            return_mask_output=True
        )
        if anchor_scores is None:
            anchors = model.map_function(model.emo_anchor)
            anchor_scores = model.score_func(
                mask_mapped_output.unsqueeze(1),
                anchors.unsqueeze(0)
            )
        sims = anchor_scores.squeeze(0).cpu().tolist()

    return [1.0 - s for s in sims]


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_png = None
    if request.method == 'POST':
        text = request.form.get('text') or None
        submitted = [text]
        for i in range(1, 5):
            active = bool(request.form.get(f'active{i}'))
            val = request.form.get(f'val{i}') or None
            submitted.append(val if active else None)

        distances = compute_anchor_distances(submitted, app.config['ARGS'])

        fig, ax = plt.subplots()
        x = list(range(len(distances)))
        bars = ax.bar(x, distances, width=0.5)
        min_idx = distances.index(min(distances))
        bars[min_idx].set_color('green')

        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=45, ha='right')
        ax.set_xlabel('Emotion Anchors')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_png = base64.b64encode(buf.getvalue()).decode()

    return render_template('index.html', plot_png=plot_png, labels=MODAL_NAMES)


def get_parser():
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.fp16 else 'cpu')
    args.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    args.tokenizer.add_tokens(["<mask>", "[VIS]", "[AUD]", "[BIO]", "[AUS]"])
    app.config['ARGS'] = args
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
