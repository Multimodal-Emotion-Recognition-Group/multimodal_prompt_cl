import os
import numpy as np
import argparse
import time
import random
import logging
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import AutoTokenizer
from torch.optim import AdamW
import copy

from sklearn.metrics import classification_report

from dataset import DialogueDataset
from model.model import CLModel, Classifier
from model.loss import loss_function
from trainer.trainer import train_or_eval_model, retrain
from utils.data_process import *

os.environ["TOKENIZERS_PARALLELISM"] = "1"


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_paramsgroup(model, args, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params': param,
            'lr': args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay': weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str,
                        default='princeton-nlp/sup-simcse-roberta-large')  # princeton-nlp/sup-simcse-roberta-base
    parser.add_argument('--bert_dim', type=int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--mask_value', type=int, default=2, help='padding')
    parser.add_argument('--wp', type=int, default=8, help='past window size')
    parser.add_argument('--wf', type=int, default=0, help='future window size')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=256,
                        help='max content length for each text, if set to 0, then no constraint')
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='MELD', type=str, help='dataset name: IEMOCAP, MELD, EmoryNLP')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')

    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='pretrained model learning rate')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=4, metavar='E', help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    # environment
    parser.add_argument("--fp16", type=bool, default=False)  # True
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--disable_training_progress_bar", action="store_true")
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)

    # ablation study
    parser.add_argument("--disable_emo_anchor", action='store_true')
    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--disable_two_stage_training", action="store_true")
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)
    parser.add_argument("--anchor_path", type=str, default=None)

    # analysis
    parser.add_argument("--save_stage_two_cache", action="store_true")
    parser.add_argument("--save_path", default='./saved_models/', type=str)

    args = parser.parse_args()
    return args


def main(args):
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu", local_rank)

    seed_everything(args.seed)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(os.path.join(args.save_path, args.dataset_name), exist_ok=True)

    if rank == 0:
        logger = get_logger(os.path.join(args.save_path, args.dataset_name, 'logging.log'))
        logger.info(f"Running on rank 0. World size = {world_size}")
        logger.info(args)
    else:
        logger = logging.getLogger("dummy")
        logger.addHandler(logging.NullHandler())

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu", local_rank)
    if rank == 0:
        print(f"Running on device: {device}, local rank {local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens(["<mask>", "[VIS]", "[AUD]", "[BIO]", "[AUS]"])

    if args.dataset_name == "IEMOCAP":
        n_classes = 6
        target_names = ['neu', 'exc', 'fru', 'sad', 'hap', 'ang']
    elif args.dataset_name == "MELD":
        n_classes = 7
        target_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    else:
        raise ValueError('Wrong dataset name. Use IEMOCAP or MELD')

    trainset = DialogueDataset(args, dataset_name=args.dataset_name, split='train', tokenizer=tokenizer)
    # validset = DialogueDataset(args, dataset_name=args.dataset_name, split='dev', tokenizer=tokenizer)
    testset = DialogueDataset(args, dataset_name=args.dataset_name, split='test', tokenizer=tokenizer)

    train_sampler = DistributedSampler(trainset, shuffle=True)
    test_sampler = DistributedSampler(testset, shuffle=False)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler,
                              pin_memory=True, num_workers=8)
    valid_loader = DataLoader([], batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(testset, batch_size=args.batch_size, sampler=test_sampler,
                             pin_memory=True, num_workers=8)

    if rank == 0:
        print('Building model...')
    model = CLModel(args, n_classes, tokenizer)
    model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(get_paramsgroup(model.module if hasattr(model, 'module') else model, args))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)

    best_test_fscore = 0.0
    best_model = copy.deepcopy(model)

    for e in range(args.epochs):
        train_sampler.set_epoch(e)

        start_time = time.time()

        # train
        train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine = \
            train_or_eval_model(model, loss_function, train_loader, e, device, args,
                                optimizer, lr_scheduler, train=True)
        lr_scheduler.step()

        # valid
        valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _ = \
            train_or_eval_model(model, loss_function, valid_loader, e, device, args, train=False)

        # test
        test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _ = \
            train_or_eval_model(model, loss_function, test_loader, e, device, args, train=False)

        if rank == 0:
            logger.info(
                f'Epoch: {e + 1}, '
                f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_fscore: {train_fscore:.4f}, '
                f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_fscore: {valid_fscore:.4f}, '
                f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_fscore: {test_fscore:.4f}, '
                f'time: {round(time.time() - start_time, 2)} sec'
            )

            if test_fscore > best_test_fscore:
                best_test_fscore = test_fscore
                best_model = copy.deepcopy(model)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, args.dataset_name, 'model_.pkl'))
                rep = classification_report(test_label, test_pred, digits=4, target_names=target_names)

    if rank == 0:
        print("Stage 1 summary")
        print(rep)
        logger.info('Finish stage 1 training!')

    if not args.disable_two_stage_training:
        dist.barrier()

        best_model.module.load_state_dict(
            torch.load(os.path.join(args.save_path, args.dataset_name, 'model_.pkl'),
                       map_location=device)
        )

        best_model.eval()

        with torch.no_grad():
            emb_train, label_train = [], []
            for batch_id, batch in enumerate(train_loader):
                input_ids, label, vis_ids, aud_ids, bio_ids, aus_ids = batch
                label = label.to(device)

                log_prob, masked_mapped_output, _, anchor_scores = \
                    best_model(input_ids, vis_ids, aud_ids, bio_ids, aus_ids, return_mask_output=True)

                emb_train.append(masked_mapped_output.detach().cpu())
                label_train.append(label.cpu())

            emb_train = torch.cat(emb_train, dim=0)
            label_train = torch.cat(label_train, dim=0)

            emb_val, label_val = [], []
            for batch_id, batch in enumerate(valid_loader):
                input_ids, label, vis_ids, aud_ids, bio_ids, aus_ids = batch
                label = label.to(device)
                log_prob, masked_mapped_output, _, anchor_scores = \
                    best_model(input_ids, vis_ids, aud_ids, bio_ids, aus_ids, return_mask_output=True)

                emb_val.append(masked_mapped_output.detach().cpu())
                label_val.append(label.cpu())

            if len(emb_val) > 0:
                emb_val = torch.cat(emb_val, dim=0)
                label_val = torch.cat(label_val, dim=0)
            else:
                emb_val = torch.tensor([])
                label_val = torch.tensor([])

            emb_test, label_test = [], []
            for batch_id, batch in enumerate(test_loader):
                input_ids, label, vis_ids, aud_ids, bio_ids, aus_ids = batch
                label = label.to(device)
                log_prob, masked_mapped_output, _, anchor_scores = \
                    best_model(input_ids, vis_ids, aud_ids, bio_ids, aus_ids, return_mask_output=True)

                emb_test.append(masked_mapped_output.detach().cpu())
                label_test.append(label.cpu())

            emb_test = torch.cat(emb_test, dim=0)
            label_test = torch.cat(label_test, dim=0)

        if rank == 0:
            print("Embedding dataset built")

        trainset_emb = TensorDataset(emb_train, label_train)
        validset_emb = TensorDataset(emb_val, label_val)
        testset_emb = TensorDataset(emb_test, label_test)

        train_sampler_emb = DistributedSampler(trainset_emb, shuffle=True)
        valid_sampler_emb = DistributedSampler(validset_emb, shuffle=False)
        test_sampler_emb = DistributedSampler(testset_emb, shuffle=False)

        train_loader_emb = DataLoader(trainset_emb, batch_size=64, sampler=train_sampler_emb,
                                      pin_memory=True, num_workers=4)
        valid_loader_emb = DataLoader(validset_emb, batch_size=64, sampler=valid_sampler_emb,
                                      pin_memory=True, num_workers=4)
        test_loader_emb = DataLoader(testset_emb, batch_size=64, sampler=test_sampler_emb,
                                     pin_memory=True, num_workers=4)

        if args.save_stage_two_cache and rank == 0:
            import pickle
            os.makedirs("cache", exist_ok=True)
            anchors = best_model.module.map_function(best_model.module.emo_anchor)
            with open(f"./cache/{args.dataset_name}.pkl", 'wb') as f:
                pickle.dump([train_loader_emb, valid_loader_emb, test_loader_emb, anchors], f)

        dist.barrier()
        if rank == 0:
            anchors = best_model.module.map_function(best_model.module.emo_anchor)
            clf = Classifier(args, anchors).to(device)
            optimizer2 = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)

            best_valid_score = 0.0
            rep_stage2 = None

            for e in range(10):
                train_loss, train_ce_loss, train_acc, _, _, train_fscore, train_detail_f1 = \
                    retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device),
                            train_loader_emb, e, device, args, optimizer2, train=True)

                valid_loss, valid_ce_loss, valid_acc, _, _, valid_fscore, valid_detail_f1 = \
                    retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device),
                            valid_loader_emb, e, device, args, optimizer2, train=False)

                test_loss, test_ce_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1 = \
                    retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device),
                            test_loader_emb, e, device, args, optimizer2, train=False)

                logger.info(
                    f"Stage2 Epoch: {e + 1}, "
                    f"train_loss: {train_loss:.4f}, train_ce_loss: {train_ce_loss:.4f}, "
                    f"train_acc: {train_acc:.4f}, train_fscore: {train_fscore:.4f}, "
                    f"valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_fscore: {valid_fscore:.4f}, "
                    f"test_loss: {test_loss:.4f}, test_ce_loss: {test_ce_loss:.4f}, "
                    f"test_acc: {test_acc:.4f}, test_fscore: {test_fscore:.4f}"
                )

                if test_fscore > best_valid_score:
                    best_valid_score = test_fscore
                    torch.save(clf.state_dict(),
                               os.path.join(args.save_path, args.dataset_name, 'clf_.pkl'))
                    rep_stage2 = classification_report(test_label, test_pred, digits=4, target_names=target_names)

            print('Stage 2 summary')
            if rep_stage2:
                print(rep_stage2)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser()
    main(args)
