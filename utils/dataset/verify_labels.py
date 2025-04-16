def verify_labels(file_path):
    allowed_chars = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_name, label = line.strip().split('\t')
            cleaned_label = ''.join(label.split())
            if not cleaned_label:
                print(f"Empty label: {image_name}")
                continue
            for char in cleaned_label:
                if char not in allowed_chars:
                    print(f"Invalid character '{char}' in label: {cleaned_label} ({image_name})")
        print(f"All labels in {file_path} are valid.")

verify_labels('datasets/custom/train.txt')
verify_labels('datasets/custom/val.txt')