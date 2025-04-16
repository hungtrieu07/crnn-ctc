import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.dataset.custom import CustomPlateDataset
from utils.model.crnn import CRNN
from utils.model.lprnet import LPRNet
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.logger import LOGGER
from utils.converter import get_custom_plate_chars

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', help='Path to pretrained model')
    parser.add_argument('data_root', help='Path to dataset root directory')
    parser.add_argument('--not-tiny', action='store_true')
    parser.add_argument('--use-lprnet', action='store_true')
    parser.add_argument('--use-origin-block', action='store_true')
    parser.add_argument('--add-stnet', action='store_true')
    parser.add_argument('--use-lstm', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    device = select_device('', batch_size=512)
    CUSTOM_CHARS = get_custom_plate_chars()

    input_shape = (94, 24) if args.use_lprnet else (168, 48)
    dataset = CustomPlateDataset(os.path.join(args.data_root, 'images'),
                                os.path.join(args.data_root, 'val.txt'),
                                input_shape=input_shape, is_train=False)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, drop_last=False)

    model = LPRNet(in_channel=3, num_classes=len(CUSTOM_CHARS) + 1,
                   use_origin_block=args.use_origin_block, add_stnet=args.add_stnet).to(device) \
        if args.use_lprnet else \
        CRNN(in_channel=3, num_classes=len(CUSTOM_CHARS) + 1, cnn_input_height=input_shape[1],
             is_tiny=not args.not_tiny, use_gru=not args.use_lstm).to(device)
    model.load_state_dict(torch.load(args.pretrained, map_location=device))
    model.eval()

    criterion = CTCLoss(blank_label=0).to(device)
    evaluator = Evaluator(blank_label=0)

    evaluator.reset()
    for idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = dataset.convert(targets)
        with torch.no_grad():
            outputs = model(images).cpu()
        acc = evaluator.update(outputs, targets)
        LOGGER.info(f"Batch:{idx} ACC:{acc * 100:.3f}")
    acc = evaluator.result()
    LOGGER.info(f"ACC: {acc * 100:.3f}")

if __name__ == '__main__':
    main()