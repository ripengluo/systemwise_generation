# coeff_transform_paramless.py
import os, sys
import math
import torch, json, numpy as np
from utils import H5CoeffDataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset


class ParamlessTransform:
    def __init__(self, method="log1p", stat_file=None):
        assert method in {"log1p", "softsign", "arctan"}
        self.method = method

        # ---------- channel-wise μ/σ ----------
        if stat_file is not None:
            data = np.load(stat_file)
            self.mu  = torch.tensor(data["mu"])[0]     # shape [2,1,1,1,1]
            self.sig = torch.tensor(data["sig"])[0]
            #print(self.mu[0], self.sig[0])
        else:
            self.mu = self.sig = None

    # ---------- forward ----------
    def f(self, x):
        if self.method == "log1p":
            y = torch.sign(x) * torch.log1p(x.abs())
        elif self.method == "softsign":
            y = x / (1 + x.abs())
        else:  # arctan
            y = (2 / torch.pi) * torch.atan(x)
        return y

    def f_inv(self, y):
        if self.method == "log1p":
            x = torch.sign(y) * (torch.exp(y.abs()) - 1)
        elif self.method == "softsign":
            y = torch.clamp(y, -0.999999, 0.999999)
            x = y / (1 - y.abs())
        else:
            y = torch.clamp(y, -0.999999, 0.999999)
            x = torch.tan(y * torch.pi / 2)
        return x

    def forward(self, coeff):
        z = self.f(coeff)
        if self.mu is not None:
            z = (z - self.mu) / (self.sig + 1e-12)
        return z

    def inverse(self, z):
        if self.mu is not None:
            z = z * self.sig + self.mu
        x = self.f_inv(z)
        return x

class DoFTransform:
    def __init__(self, stat_file: str | None = None):
        self.mu  = None   # torch([6])
        self.sig = None   # torch([6])
        if stat_file is not None:
            self.load(stat_file)

    def to(self, device: torch.device | str):
        if self.mu  is not None: self.mu  = self.mu.to(device)
        if self.sig is not None: self.sig = self.sig.to(device)
        return self

    @torch.no_grad()
    def fit(self, loader, dof_index: int = 2, device: str | torch.device = "cuda"):
        sum_ = torch.zeros(6, device=device)
        sq_  = torch.zeros(6, device=device)
        cnt  = torch.zeros(6, device=device)

        for batch in loader:
            dof  = batch[dof_index].to(device).float()
            #mask = batch[mask_index].to(device).float()
            sum_ += (dof).sum(dim=0)
            sq_  += ((dof * dof)).sum(dim=0)
            #cnt  += mask.sum(dim=0)
            cnt  += dof.size(0)

        mu = sum_ / cnt.clamp_min(1.0)
        var = (sq_ / cnt.clamp_min(1.0)) - mu * mu
        sig = var.clamp_min(1e-8).sqrt()
        sig = torch.where(sig < 1e-6, torch.ones_like(sig), sig)

        self.mu, self.sig = mu.detach(), sig.detach()
        return self.mu, self.sig

    @torch.no_grad()
    def forward(self, dof: torch.Tensor): #, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mu is not None:
            dof = (dof - self.mu) / self.sig
        #if mask is not None:
        #    dof = dof * mask
        return dof

    @torch.no_grad()
    def inverse(self, z_norm: torch.Tensor) -> torch.Tensor:
        z = z_norm
        if self.mu is not None:
            z = z * self.sig + self.mu
        return z

    def save(self, path: str):
        assert self.mu is not None and self.sig is not None, "Call fit(...) or load(...) first."
        np.savez(path,
                 dof_mu=self.mu.detach().cpu().numpy(),
                 dof_sig=self.sig.detach().cpu().numpy())

    def load(self, path: str):
        data = np.load(path)
        mu  = data["dof_mu"] if "dof_mu" in data else data["mu"]
        sig = data["dof_sig"] if "dof_sig" in data else data["sig"]
        self.mu  = torch.tensor(mu,  dtype=torch.float32)
        self.sig = torch.tensor(sig, dtype=torch.float32)
        self.sig = torch.where(self.sig < 1e-6, torch.ones_like(self.sig), self.sig)
        return self

def main():
    chg_dim = 16
    data_dir = "./dataset"
    #h5_paths = [os.path.join(data_dir, f"mp_chg_{i:03d}.h5") for i in range(8)]
    h5_paths = [os.path.join(data_dir, f"mp_chg_{i:03d}.h5") for i in [0,1,2,4, 5, 7, 8, 9, 10, 11, 12]]
    dataset = ConcatDataset([H5CoeffDataset(k, chg_dim) for k in h5_paths])
    #dataset = H5CoeffDataset("dataset/mp_chg_all_prop.h5", chg_dim)

    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,pin_memory=True)
    
    tx = ParamlessTransform(method="log1p")

    CHG_DIM = 16
    sum_c     = torch.zeros(2)
    sumsq_c   = torch.zeros(2)
    num_voxels= 0
    
    with torch.no_grad():
        for coeff, *_ in loader:        # coeff: [B,2,16,16,16]
            z = tx.f(coeff)             # log1p / softsign / arctan
    
            sum_c   += z.sum(dim=(0,2,3,4))          # shape [2]
            sumsq_c += (z**2).sum(dim=(0,2,3,4))     # shape [2]
    
            num_voxels += z.size(0) * CHG_DIM**3
    
    mu    = sum_c   / num_voxels                 # shape [2]
    var   = sumsq_c / num_voxels - mu**2
    sigma = torch.sqrt(var + 1e-12)
    
    print("μ:", mu.tolist())
    print("σ:", sigma.tolist())
    
    np.savez(
        "paramless_stats.npz",
        mu   = mu.view(2,1,1,1,1).cpu().numpy(),
        sig  = sigma.view(2,1,1,1,1).cpu().numpy(),
        meth = np.array("log1p"),
    )
    return

def main2():
    device = "cuda"
    chg_dim = 16
    data_dir = "./dataset"
    #h5_paths = [os.path.join(data_dir, f"mp_chg_{i:03d}.h5") for i in range(8)]
    h5_paths = [os.path.join(data_dir, f"mp_chg_{i:03d}.h5") for i in [0,1,2,4, 5, 7, 8, 9, 10, 11, 12]]
    dataset = ConcatDataset([H5CoeffDataset(k, chg_dim) for k in h5_paths])
    #dataset = H5CoeffDataset("dataset/mp_chg_all_prop.h5", chg_dim)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,pin_memory=True)

    tx = DoFTransform().to(device)
    mu, sig = tx.fit(loader, device=device)
    tx.save("./dof_stats")
    print("mu:", mu.tolist())
    print("sigma:", sig.tolist())
    return


if __name__ == "__main__":
    main()
    main2()
