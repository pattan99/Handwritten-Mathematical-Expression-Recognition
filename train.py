from logger import Logger
logger = Logger("train_official_ctc_continue.log")

import os
MATHWRITING_ROOT_DIR='data/mathwriting-2024'
TRAIN_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'train')
VAL_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'valid')
TEST_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'test')
SYMBOL_DIR = os.path.join(MATHWRITING_ROOT_DIR, 'symbols')

import pandas as pd
train_df = pd.read_csv("data/train.csv")
train_df['file_path'] = train_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/'+x.split('/')[-2]+'/'+x.split('/')[-1].replace('.inkml','.bin'))
train_df['image_path'] = train_df['file_path'].apply(lambda x: x.replace('train', 'train_img').replace('.bin','.png'))
# train_df['file_path'] = train_df['file_path'].apply(lambda x: x.replace('train', 'train_merged'))

valid_df = pd.read_csv("data/val.csv")
valid_df['file_path'] = valid_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/'+x.split('/')[-2]+'/'+x.split('/')[-1].replace('.inkml','.bin'))
valid_df['image_path'] = valid_df['file_path'].apply(lambda x: x.replace('valid', 'valid_img').replace('.bin','.png'))
valid_df['file_path'] = valid_df['file_path'].apply(lambda x: x.replace('valid', 'val'))

test_df = pd.read_csv("data/test.csv")
test_df['file_path'] = test_df['file_path'].apply(lambda x: MATHWRITING_ROOT_DIR+'/'+x.split('/')[-2]+'/'+x.split('/')[-1].replace('.inkml','.bin'))
test_df['image_path'] = test_df['file_path'].apply(lambda x: x.replace('test', 'test_img').replace('.bin','.png'))
# test_df['file_path'] = test_df['file_path'].apply(lambda x: x.replace('test', 'test_merged'))

# removed_file_paths = [
#     "data/mathwriting-2024/train_merged/959bf7ff999b61d8.npy",
#     "data/mathwriting-2024/train_merged/b8303e072af71d4e.npy",
#     "data/mathwriting-2024/train_merged/96c224cb52c3b084.npy",
#     "data/mathwriting-2024/train_merged/3e43b67ceb3b5947.npy",
#     "data/mathwriting-2024/train_merged/5e1cdcb62a890d2e.npy",
#     "data/mathwriting-2024/train_merged/c30ab69f39adbda3.npy",
#     "data/mathwriting-2024/train_merged/b9d82762a2a7dc60.npy",
# ]

removed_file_paths = [
    "data/mathwriting-2024/train/a21ff5968b022586.bin",
    "data/mathwriting-2024/train/f13fe0616d6ba693.bin",
    "data/mathwriting-2024/train/2d6b345c1f786c82.bin",
]

train_df = train_df[~train_df['file_path'].isin(removed_file_paths)]

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
from dataloader import StrokeDataset, collate_fn

train_dataset = StrokeDataset(data_dir=MATHWRITING_ROOT_DIR, labels_df=train_df)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
valid_dataset = StrokeDataset(data_dir=MATHWRITING_ROOT_DIR, labels_df=test_df)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import init_clip_model, load_clip_model
from model import OnHWRTransformer
from trainer import train, evaluate

clip_model = init_clip_model(device)
clip_model = load_clip_model('checkpoints/latest_line_1_patch_16_9.pt', device)

num_encoder_layers = 11
nhead = 8
d_model = 256
dim_feedforward = 512

start_epoch = 0
end_epoch = 50
vocab_size = 257

model = OnHWRTransformer(vocab_size=vocab_size, clip_model=clip_model, d_model=d_model, nhead=nhead, 
                         num_encoder_layers=num_encoder_layers, 
                         dim_feedforward=dim_feedforward, dropout=0.1).to(device)

# checkpoint = torch.load(f'checkpoints_onhmer/checkpoint_{num_encoder_layers}_{nhead}_{dim_feedforward}_{start_epoch}.pth',
#                         map_location=device)

# model.load_state_dict(checkpoint['model'])

optimizer = optim.AdamW(model.parameters(), lr=0.0005)
# Use sgd
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)

#optimizer.load_state_dict(checkpoint['optimizer'])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch - start_epoch, eta_min=0.000005)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # 0 is usually the blank index

for epoch in range(start_epoch, end_epoch):
    logger.info(f'Running epoch {epoch + 1}, lr: {optimizer.param_groups[0]["lr"]}')
    
    train_loss = train(model, train_loader, optimizer, criterion, device, logger, debug=False, verbose_freq = 8000)
    valid_cer = evaluate(model, valid_loader, device)

    scheduler.step()
    
    checkpoint = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, f'checkpoints_onhmer/checkpoint_{num_encoder_layers}_{nhead}_{dim_feedforward}_{epoch+1}.pth')
    
    logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid CER: {valid_cer}')