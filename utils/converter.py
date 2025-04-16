from utils.logger import LOGGER
import torch

def get_custom_plate_chars():
    # Digits (0-9), uppercase letters (A-Z), and special characters (., -)
    digits = list('0123456789')
    letters = list('ABCDEFGHIJKLMNPQRSTUVWXYZ')
    special = ['.', '-']
    return digits + letters + special  # 38 characters

class StrLabelConverter:
    def __init__(self, alphabet=None):
        if alphabet is None:
            alphabet = get_custom_plate_chars()
        self.alphabet = alphabet + ['_']  # Use '_' as blank for CTC
        self.char2idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.char2idx['_'] = 0  # Blank token for CTC

    def encode(self, text):
        try:
            indices = [self.char2idx[char] for char in text]
            return torch.IntTensor(indices), torch.IntTensor([len(indices)])
        except KeyError as e:
            LOGGER.error(f"Character '{e}' not found in alphabet: {text}")
            raise

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length.item()
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] if i > 0 else '_' for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (i == 0 or t[i] != t[i - 1]):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            for i in range(length.size(0)):
                texts.append(self.decode(t[i], length[i], raw))
            return texts