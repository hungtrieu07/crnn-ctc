# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午4:42
@file: eval.py
@author: zj
@description:

Usage - Single-GPU eval:
    $ python eval_emnist.py runs/emnist_ddp/crnn-emnist-e100.pth ../datasets/emnist/
    $ python eval_emnist.py runs/emnist_ddp/crnn-emnist-e100.pth ../datasets/emnist/ --use-gru

"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.model.crnn import CRNN
from utils.dataset.emnist import EMNISTDataset, DIGITS_CHARS
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('--use-gru', action='store_true', help='use nn.GRU instead of nn.LSTM')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(args, val_root, pretrained):
    model = CRNN(in_channel=1, num_classes=len(DIGITS_CHARS), cnn_output_height=1, use_gru=args.use_gru)
    print(f"Loading CRNN pretrained: {pretrained}")
    ckpt = torch.load(pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    img_h = 32
    digits_per_sequence = 5
    val_dataset = EMNISTDataset(val_root, is_train=False, num_of_sequences=50000,
                                digits_per_sequence=digits_per_sequence, img_h=img_h)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)

    blank_label = len(DIGITS_CHARS) - 1
    emnist_evaluator = Evaluator(blank_label=blank_label)

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).cpu()

        acc = emnist_evaluator.update(outputs, targets)
        info = f"Batch:{idx} ACC:{acc * 100:.3f}"
        pbar.set_description(info)
    acc = emnist_evaluator.result()
    print(f"ACC:{acc * 100:.3f}")


def main():
    args = parse_opt()

    val(args, args.val_root, args.pretrained)


if __name__ == '__main__':
    main()
