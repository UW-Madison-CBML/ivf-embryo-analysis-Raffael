# model_conv_lstm_ae.py
import torch, torch.nn as nn

class FrameEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,128,1,1]
        )
        self.proj = nn.Linear(128, out_dim)
    def forward(self, x):             # x: [B,1,128,128]
        h = self.net(x).squeeze(-1).squeeze(-1)  # [B,128]
        return self.proj(h)           # [B,out_dim]

class FrameDecoder(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, 8*8*128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),   # 32x32
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),   # 64x64
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid()  # 128x128
        )
    def forward(self, z):             # z: [B,128]
        x = self.fc(z).view(-1,128,8,8)
        return self.deconv(x)         # [B,1,128,128]

class ConvLSTMAE(nn.Module):
    def __init__(self, emb=128, lstm_hid=128):
        super().__init__()
        self.enc = FrameEncoder(out_dim=emb)
        self.lstm_enc = nn.LSTM(input_size=emb, hidden_size=lstm_hid, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=lstm_hid, hidden_size=lstm_hid, batch_first=True)
        self.dec = FrameDecoder(in_dim=lstm_hid)

    def forward(self, vol):           # vol: [B,T,1,128,128]
        B,T,_,_,_ = vol.shape
        f = self.enc(vol.view(B*T,1,128,128))     # [B*T,emb]
        f = f.view(B,T,-1)                        # [B,T,emb]
        z_seq, _ = self.lstm_enc(f)               # [B,T,lstm_hid]
        # 解碼：逐幀
        h_dec, _ = self.lstm_dec(z_seq)           # [B,T,lstm_hid]
        recon = []
        for t in range(T):
            recon.append(self.dec(h_dec[:,t,:]))  # [B,1,128,128]
        recon = torch.stack(recon, dim=1)         # [B,T,1,128,128]
        return recon, z_seq                       # recon, latent per frame

