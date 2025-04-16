import argparse
import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from utils.dataset.custom import CustomPlateDataset
from utils.model.crnn import CRNN
from utils.model.lprnet import LPRNet
from utils.converter import StrLabelConverter, get_custom_plate_chars
from utils.torchutil import select_device
from utils.logger import LOGGER

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', help='Path to pretrained model')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('save_dir', help='Path to save predictions')
    parser.add_argument('--not-tiny', action='store_true')
    parser.add_argument('--use-lprnet', action='store_true')
    parser.add_argument('--use-origin-block', action='store_true')
    parser.add_argument('--add-stnet', action='store_true')
    parser.add_argument('--use-lstm', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    device = select_device('', batch_size=1)
    CUSTOM_CHARS = get_custom_plate_chars()
    converter = StrLabelConverter()

    input_shape = (94, 24) if args.use_lprnet else (168, 48)
    model = LPRNet(in_channel=3, num_classes=len(CUSTOM_CHARS) + 1,
                   use_origin_block=args.use_origin_block, add_stnet=args.add_stnet).to(device) \
        if args.use_lprnet else \
        CRNN(in_channel=3, num_classes=len(CUSTOM_CHARS) + 1, cnn_input_height=input_shape[1],
             is_tiny=not args.not_tiny, use_gru=not args.use_lstm).to(device)
    model.load_state_dict(torch.load(args.pretrained, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[0])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        preds = model(image)
        preds_size = torch.IntTensor([preds.size(0)]).to(device)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        pred_text = converter.decode(preds.data, preds_size.data, raw=False)
    predict_time = (time.time() - t0) * 1000
    LOGGER.info(f"Pred: {pred_text} - Predict time: {predict_time:.1f} ms")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"plate_{os.path.basename(args.image_path)}")
    with open(save_path, 'w') as f:
        f.write(pred_text)
    LOGGER.info(f"Save to {save_path}")

if __name__ == '__main__':
    main()