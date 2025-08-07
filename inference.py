import time
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer
from models.blip import blip_decoder

# ========== CONFIG ==========

class Args:
    image_size = 224
    load_pretrained = '/kaggle/input/your_model_path/model_best.pth'  # <== Change to your .pth file
    device = 'cpu'  # or 'cuda' if available
    cls_weight = 4
    clip_k = 21
args = Args()

# ========== SETUP ==========

# Device
device = torch.device(args.device)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({'bos_token': '[DEC]'})
tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

# Prompt and model
labels_temp = ['[BLA]'] * 18
prompt_temp = ' '.join(labels_temp) + ' '
model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)
model.to(device)
model.eval()

# Load weights
if args.load_pretrained:
    state_dict = torch.load(args.load_pretrained, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from {args.load_pretrained}")

# ========== DUMMY INPUT ==========

batch_size = 1
dummy_image = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
dummy_caption = ['this is a dummy caption'] * batch_size
dummy_cls_labels = torch.zeros(batch_size, 18, dtype=torch.long).to(device)
dummy_clip_memory = torch.randn(batch_size, args.clip_k, 512).to(device)
criterion_cls = nn.CrossEntropyLoss()
dummy_base_probs = np.ones(18, dtype=np.float32)

# ========== WARMUP ==========

with torch.no_grad():
    for _ in range(3):
        _ = model(dummy_image, dummy_caption, dummy_cls_labels, dummy_clip_memory, criterion_cls, dummy_base_probs)

# ========== INFERENCE TIMING ==========

with torch.no_grad():
    start = time.time()
    _ = model(dummy_image, dummy_caption, dummy_cls_labels, dummy_clip_memory, criterion_cls, dummy_base_probs)
    end = time.time()

print(f"Inference Time (1 sample): {end - start:.6f} seconds")
