import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pickle
from utils.data_process import *

class DialogueDataset(Dataset):    
    def __init__(self, args, dataset_name = 'IEMOCAP', split = 'train', speaker_vocab=None, label_vocab=None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.split = split
        # if osp.exists(self.save_path(dataset_name)):
        #     self.data, self.labels = torch.load(osp.join(self.save_path, f"{split}.pt"))
        # else:
        self.tokenizer = tokenizer
        self.wp = args.wp
        self.wf = args.wf
        self.max_len = args.max_len
        self.pad_value = args.pad_value
        self.dataset_name = dataset_name

        self.emotion_map = pickle.load(open(f'./data/{dataset_name}/label_vocab.pkl', 'rb'))
        # print(self.emotion_map)
        self.emotion_map = {v:k for k,v in self.emotion_map.items()}
        _special_tokens_ids = tokenizer('<mask>')['input_ids']
        self.CLS = _special_tokens_ids[0]
        self.MASK = _special_tokens_ids[1]
        self.SEP = _special_tokens_ids[2]
        self.VIS = self.MASK + 1
        self.AUD = self.MASK + 2
        self.BIO = self.MASK + 3
        self.AUS = self.MASK + 4

        self.data, self.labels, self.utterance_sequence = self.read(dataset_name, split, tokenizer)
        
        assert len(self.data) == len(self.labels)

    def pad_to_len(self, list_data, max_len, pad_value):
        list_data = list_data[-max_len:]
        len_to_pad = max_len - len(list_data)
        pads = [pad_value] * len_to_pad
        list_data.extend(pads)
        return list_data

    def read(self, dataset_name, split, tokenizer):
        if dataset_name == "IEMOCAP":
            dialogs = load_iemocap_turn(f'./data/{dataset_name}/modified_{split}_data.json')
        elif dataset_name == "EmoryNLP":
            dialogs = load_emorynlp_turn(f'./data/{dataset_name}/{split}_data.json')
        elif dataset_name == "MELD":
            dialogs = load_meld_turn(f'./data/{dataset_name}/modified4_{split}_data.csv')
        print("number of dialogs:", len(dialogs))

        data_list = []
        label_list = []
        utterance_sequence = []
        ret_utterances = []
        ret_labels = []


        for dialogue in dialogs:
            utterance_ids = []
            utterance_seq = []
            for idx, turn_data in enumerate(dialogue):
                text_with_speaker = turn_data['speaker'] + ' says: ' + turn_data['text']
                token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
                utterance_ids.append(token_ids)
                if turn_data['label'] < 0:
                    continue
                full_context = [self.CLS, self.VIS, self.AUD, self.BIO, self.AUS, self.SEP]
                lidx = 0
                for lidx in range(idx):
                    total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 12 # 8
                    if total_len + len(utterance_ids[idx]) <= self.max_len:
                        break
                lidx = max(lidx, idx-12) # 8
                for item in utterance_ids[lidx:]:
                    full_context.extend(item)
                
                query_idx = idx
                input_ids = full_context[:-len(utterance_ids[query_idx])]
                ret_utterances.append(
                    (input_ids, 
                     turn_data['speaker'], 
                     turn_data['text'], 
                     turn_data['vis_cap'], 
                     turn_data['audio_cap'],
                     turn_data['bio'],
                     turn_data['aus'])
                )# input_ids, speaker
                ret_labels.append(dialogue[query_idx]['label'])

                utterance_seq.append({
                    "uttrance": text_with_speaker,
                    "emotion": dialogue[query_idx]['label']
                })
                utterance_sequence.append(utterance_seq + [])

        data_list = ret_utterances
        label_list = torch.LongTensor(ret_labels)
        return data_list, label_list, utterance_sequence

    def process(self, data):
        input_ids, speaker, text, vis_cap, aud_cap, bio, aus = data
        # print(input_ids)
        p2 = 'For utterance: '+ text + " " + speaker + " feels <mask> "
        p2 = self.tokenizer(p2)['input_ids'][1:]
        p2 = input_ids + p2
        
        vis_ids = self.tokenizer(vis_cap)['input_ids']
        aud_ids = self.tokenizer(aud_cap)['input_ids']
        bio_ids = self.tokenizer(bio)['input_ids']
        aus_ids = self.tokenizer(aus)['input_ids']

        p2 = pad_to_len(p2, self.max_len, self.pad_value)
        p2 = torch.LongTensor(p2)
        vis_ids = pad_to_len(vis_ids, self.max_len, self.pad_value)
        vis_ids = torch.LongTensor(vis_ids)
        aud_ids = pad_to_len(aud_ids, self.max_len, self.pad_value)
        aud_ids = torch.LongTensor(aud_ids)
        bio_ids = pad_to_len(bio_ids, self.max_len, self.pad_value)
        bio_ids = torch.LongTensor(bio_ids)
        aus_ids = pad_to_len(aus_ids, self.max_len, self.pad_value)
        aus_ids = torch.LongTensor(aus_ids)

        return p2, vis_ids, aud_ids, bio_ids, aus_ids

    def save_path(self, dataset_name):
        return f'./data/{dataset_name}/processed/{self.split}'

    def __getitem__(self, index):
        text = self.data[index]
        text, vis_ids, aud_ids, bio_ids, aus_ids = self.process(text)
        label = self.labels[index]
       
        return text, label, vis_ids, aud_ids, bio_ids, aus_ids

    def __len__(self):
        return len(self.data)