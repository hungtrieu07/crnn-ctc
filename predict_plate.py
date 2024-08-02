# -*- coding: utf-8 -*-

"""
@date: 2023/10/10 上午9:59
@file: predict.py
@author: zj
@description:

Usage: Predict Plate:
    $ python predict_plate.py runs/plate_ddp/crnn-plate-e100.pth ./assets/plate/宁A87J92_0.jpg runs/
    $ python predict_plate.py runs/plate_ddp/crnn-plate-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/

"""

import argparse
import os
from itertools import groupby

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# cp assets/fonts/simhei.ttf /usr/share/fonts/truetype/noto/
# rm -rf ~/.cache/matplotlib/*
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

import importlib

# 根据脚本是否作为主模块运行来决定导入方式
if __name__ == '__main__':
    # 直接运行时，使用绝对导入
    CRNN = importlib.import_module('utils.model.crnn').CRNN
    PLATE_CHARS = importlib.import_module('utils.dataset.plate').PLATE_CHARS
else:
    # 被导入时，尝试使用相对导入，如果失败则回退到绝对导入
    try:
        CRNN = importlib.import_module('.utils.model.crnn', package=__package__).CRNN
        PLATE_CHARS = importlib.import_module('.utils.dataset.plate', package=__package__).PLATE_CHARS
    except ValueError:
        CRNN = importlib.import_module('utils.model.crnn').CRNN
        PLATE_CHARS = importlib.import_module('utils.dataset.plate').PLATE_CHARS


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict CRNN with EMNIST')
    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, help='path to pretrained model')
    parser.add_argument('image_path', metavar='IMAGE', type=str, help='path to image path')
    parser.add_argument('save_dir', metavar='DST', type=str, help='path to save dir')

    parser.add_argument('--use-gru', action='store_true', help='use nn.GRU instead of nn.LSTM')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


def load_model(pretrained=None, device=None, use_gru=False):
    model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_output_height=2, use_gru=use_gru)
    if pretrained is not None:
        if isinstance(pretrained, list):
            pretrained = pretrained[0]
        print(f"Loading CRNN pretrained: {pretrained}")
        ckpt = torch.load(pretrained, map_location='cpu')
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


@torch.no_grad()
def predict(image, model=None, device=None):
    # Data
    img_w = 168
    img_h = 48
    resize_image = cv2.resize(image, (img_w, img_h))

    data = torch.from_numpy(resize_image).float() / 255.
    # HWC -> CHW
    data = data.permute(2, 0, 1)

    # Infer
    # [1, H, N*W] -> [1, 1, H, N*W]
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(data).cpu()[0]

    _, max_index = torch.max(output, dim=1)
    raw_pred = list(max_index.numpy())
    blank_label = 0
    pred = torch.IntTensor([c for c, _ in groupby(raw_pred) if c != blank_label])
    pred = pred.numpy()

    pred_plate = [PLATE_CHARS[i] for i in pred]
    # pred_plate = ''.join(pred_plate)
    pred_plate = ''.join(pred_plate[:2]) + "·" + ''.join(pred_plate[2:])

    return pred_plate


def main():
    args = parse_opt()

    image_path = args.image_path
    assert os.path.isfile(image_path), image_path
    image = cv2.imread(image_path)
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Model
    model, device = load_model(pretrained=args.pretrained, use_gru=args.use_gru)

    # Predict
    pred_plate = predict(image=image, model=model, device=device)

    # Draw
    plt.figure()
    title = f"Pred: {pred_plate}"
    print(title)

    plt.title(title)
    plt.imshow(image)
    plt.axis('off')

    # Save
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    res_path = os.path.join(save_dir, f"plate_{image_name}")
    print(f'Save to {res_path}')
    plt.savefig(res_path)


if __name__ == '__main__':
    main()
