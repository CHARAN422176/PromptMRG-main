import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.tester import Tester
from models.blip import blip_decoder
from dataset import create_dataset_test 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from transformers import BertTokenizer 


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings 
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed     training')
    parser.add_argument('--device', default='cuda')

    # cls head
    parser.add_argument('--cls_weight', type=float, default=4, help='Loss weight of classification branch.')
    parser.add_argument('--clip_k', type=int, default=21, help='Number of retrieved reports from database.')

    args = parser.parse_args()
    return args


import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.tester import Tester
from models.blip import blip_decoder
from dataset import create_dataset_test 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from transformers import BertTokenizer 


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings 
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed     training')
    parser.add_argument('--device', default='cuda')

    # cls head
    parser.add_argument('--cls_weight', type=float, default=4, help='Loss weight of classification branch.')
    parser.add_argument('--clip_k', type=int, default=21, help='Number of retrieved reports from database.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    utils.init_distributed_mode(args)  # from blip
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank() # from blip
    torch.manual_seed(args.seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])
    print("Tokenizer size:", len(tokenizer))  # Should be 30527

    # build model architecture
    labels_temp = ['[BLA]'] * 18
    prompt_temp = ' '.join(labels_temp) + ' '
    model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)

    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.load_pretrained))

    # --- Parameter counting ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- FLOPs calculation section ---
    # Install thop if not present
    try:
        from thop import profile, clever_format
    except ImportError:
        import os
        os.system('pip install -q thop')
        from thop import profile, clever_format

    # Prepare dummy inputs that match the model's forward
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, args.image_size, args.image_size)
    dummy_caption = ['this is a dummy caption'] * batch_size
    dummy_cls_labels = torch.zeros(batch_size, 18, dtype=torch.long)
    clip_k = 21
    clip_dim = 512
    dummy_clip_memory = torch.randn(batch_size, clip_k, clip_dim)
    criterion_cls = nn.CrossEntropyLoss()
    dummy_base_probs = np.ones(18, dtype=np.float32)

    # Patch forward if needed (for torchscript/thop tracing)
    orig_forward = model.forward
    def patched_forward(image, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        with torch.no_grad():
            return orig_forward(image, caption, cls_labels, clip_memory, criterion_cls, base_probs)
    model.forward = patched_forward

    macs, params = profile(
        model,
        inputs=(
            dummy_image,
            dummy_caption,
            dummy_cls_labels,
            dummy_clip_memory,
            criterion_cls,
            dummy_base_probs
        )
    )
    macs_cf, params_cf = clever_format([macs, params], "%.3f")
    print(f"FLOPs (thop): {macs_cf}")
    print(f"Parameters (thop): {params_cf}")

    model.forward = orig_forward  # Restore original if needed

if __name__ == '__main__':
    main()
