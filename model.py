from CLIP.clip.model import *
from utils import VectorizeCharCLIP

def init_clip_model(device):
    vc = VectorizeCharCLIP(max_len=100)
    context_length = 102

    clip_model = CLIP(embed_dim=512,
                image_resolution=224,
                vision_layers=12,
                vision_width=768,
                vision_patch_size=16,
                context_length=context_length,
                vocab_size=len(vc.vocab),
                transformer_width=512,
                transformer_heads=512,
                transformer_layers=12).to(device)

    width = 768
    scale = width ** -0.5
    patch_size = 16
    clip_model.visual.positional_embedding = nn.Parameter(scale * torch.randn((224 // patch_size) * (144 // patch_size) + 1, width).to(device))

    return clip_model

def load_clip_model(checkpoint_path, device):
    clip_model = init_clip_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    clip_model.load_state_dict(checkpoint['model'])
    if device == "cpu":
        pass
    else :
        convert_weights(clip_model)
    return clip_model


import torch
import torch.nn as nn
import math

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)
        
class OnHWRTransformer(nn.Module):
    def __init__(self, vocab_size, clip_model, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.clip_model = clip_model
        #self.clip_proj = nn.Linear(self.clip_model.visual.output_dim, d_model)
        self.clip_proj = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, d_model),
            nn.LayerNorm(d_model)
        )
        # Giữ nguyên Conv2D layers
        self.conv1 = Conv2DBlock(1, 20)
        self.conv2 = Conv2DBlock(20, 20)
        
        # Input có width=20, project thẳng lên d_model
        #self.linear_proj = nn.Linear(20 * 20, d_model)
        self.linear_proj = nn.Sequential(
            nn.Linear(20 * 20, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_emb = nn.Embedding(1400, d_model)
        self.positional_layernorm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, img):
        # src: [batch_size, 1, seq_len, 20]
        x = self.conv1(src)  # [batch_size, 20, seq_len, 20]
        x = self.conv2(x)    # [batch_size, 20, seq_len, 20]
        
        # Permute để seq_len thành chiều thứ 2
        x = x.permute(0, 2, 3, 1)  # [batch_size, seq_len, 20, 20]
        b, seq_len, w, c = x.shape
        x = x.reshape(b, seq_len, -1)  # [batch_size, seq_len, 20]
        
        x = self.linear_proj(x)  # [batch_size, seq_len, d_model]
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img)
        image_features = image_features.float()
        clip_features = self.clip_proj(image_features).unsqueeze(1)  # [batch_size, 1, d_model]

        x = torch.cat([clip_features, x], dim=1)  # [batch_size, seq_len+1, d_model]
        x = x.permute(1, 0, 2)  # [seq_len+1, batch_size, d_model]

        # Position embedding trên độ dài thực tế của chuỗi
        seq_len = x.size(0)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
        x = x + self.pos_emb(positions)
        x = self.positional_layernorm(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)  # [seq_len+1, batch_size, vocab_size]
        output = x.log_softmax(-1)  # Thêm softmax trước CTC
        
        return output

    def predict(self, src, img):
        """
        Thực hiện inference và giải mã kết quả.

        Args:
            src: Tensor đầu vào.
            img: Tensor ảnh.
            vocab: Danh sách bộ từ vựng.

        Returns:
            Danh sách các chuỗi dự đoán (kiểu string).
        """
        self.eval() # Đảm bảo model ở chế độ eval
        with torch.no_grad():
            output = self(src, img)  # [seq_len + 1, batch_size, vocab_size]

        # Giải mã bằng argmax
        predicted_indices = torch.argmax(output, dim=-1)

        # Giải mã CTC cho từng chuỗi trong batch
        decoded_sequences = []
        for batch_idx in range(predicted_indices.size(1)):
            decoded_sequence = []
            for timestep in predicted_indices[:, batch_idx]:
                if timestep != 0 and (not decoded_sequence or timestep != decoded_sequence[-1]):
                    decoded_sequence.append(timestep.item())
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences