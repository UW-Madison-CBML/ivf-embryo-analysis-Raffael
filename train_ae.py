# train_ae.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from model_conv_lstm_ae import ConvLSTMAE
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    ds = IVFSequenceDataset("index.csv", resize=128, norm="minmax01")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    model = ConvLSTMAE(emb=128, lstm_hid=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    l1 = nn.L1Loss()

    for epoch in range(20):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        for vol, _ in pbar:
            vol = vol.to(DEVICE)                         # [B,T,1,128,128]
            recon, z_seq = model(vol)
            rec_loss = l1(recon, vol)
            smooth = ((z_seq[:,1:]-z_seq[:,:-1])**2).mean()  # temporal smooth
            loss = rec_loss + 0.1 * smooth
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", sm=f"{smooth.item():.4f}")
        print(f"epoch {epoch} avg loss={total/len(loader):.4f}")
        torch.save(model.state_dict(), f"ae_epoch{epoch}.pt")

if __name__ == "__main__":
    train()

