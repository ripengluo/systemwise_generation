import sys
import math, json
import numpy as np
from itertools import permutations, product
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import torchdiffeq, torchsde
from torch import distributed as dist
from torchdyn.core import NeuralODE

import h5py
from scipy.fftpack import idctn
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Chgcar
import json, copy



class H5CoeffDataset(Dataset):
    """`h5py.File` cannot be pickled (breaks DataLoader workers).
    Keep only the file‑path; each process/worker lazily opens its own handle.
    """
    def __init__(self, h5_path: str, chg_dim: int = 16):
        self.h5_path = h5_path
        self.chg_dim = chg_dim
        # Pre‑scan keys once (main process) so that workers share the list
        with h5py.File(h5_path, "r") as f:
            self.keys = [k for k in f if min(f[k]["coeff"].shape[1:]) >= chg_dim]
        self._h5 = None  # will be opened on first __getitem__

    @property
    def h5(self):
        if self._h5 is None:
            # readonly = default → safe for many processes
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    # Pickle‑safe -----------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None            # drop open file handle before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # POSCAR cell parser ---------------------------------------------
    @staticmethod
    def _get_cell(poscar: str):
        lines = poscar.strip().splitlines()
        scale = float(lines[1].split()[0])
        lattice = [list(map(float, lines[i].split()[:3])) for i in range(2, 5)]
        return scale * torch.tensor(lattice)

    def one_hot_encoder(self, lat_sys):
        sys_oh = torch.zeros(7)
        # logV, log(a/b), log(c/b), alpha, beta, gamma
        if lat_sys == "cubic":
            sys_oh[0] = 1
            DoF_mask = torch.tensor([1,0,0,0,0,0]) # logV, 0, 0, 90, 90, 90
        elif lat_sys == "tetragonal":
            sys_oh[1] = 1
            DoF_mask = torch.tensor([1,0,1,0,0,0]) # logV, 0, log(c/a), 90, 90, 90
        elif lat_sys == "orthorhombic":
            sys_oh[2] = 1
            DoF_mask = torch.tensor([1,1,1,0,0,0]) # logV, log(a/b), log(c/b), 90, 90, 90
        elif lat_sys == "hexagonal":
            sys_oh[3] = 1
            DoF_mask = torch.tensor([1,0,1,0,0,0]) # logV, 0, log(c/a), 90, 90, 120
        elif lat_sys == "trigonal":
            sys_oh[4] = 1
            DoF_mask = torch.tensor([1,0,1,0,0,0]) # logV, 0, log(c/a), 90, 90, 120 
        elif lat_sys == "monoclinic":
            sys_oh[5] = 1
            DoF_mask = torch.tensor([1,1,1,0,0,1]) # logV, log(a/b), log(c/b), 90, 90, gamma
        elif lat_sys == "triclinic":
            sys_oh[6] = 1
            DoF_mask = torch.tensor([1,1,1,1,1,1]) # logV, log(a/b), log(c/b), alpha, beta, gamma
        else:
            print(lat_sys)
            raise TypeError("Ambiguous lattice system is given, must in the catagory of 7 lattice systems!")
        return sys_oh, DoF_mask


    def theta_to_z(self, theta_rad):
        EPS = 1e-4
        t = torch.as_tensor(theta_rad, dtype=torch.float32)
        u = 2.0 * t / math.pi - 1.0
        u = u.clamp(min=-1.0 + EPS, max=1.0 - EPS)
        z = 0.5 * torch.log((1 + u) / (1 - u))
        return z if isinstance(theta_rad, torch.Tensor) else float(z)

    def get_dof(self, conv, debug=False): # here conv == stru, since conv is stored when importing data

        vol = conv.volume
        lengths = conv.lattice.lengths
        angles = conv.lattice.angles

        log_a_b = math.log(lengths[0]/lengths[1])
        log_c_b = math.log(lengths[2]/lengths[1])

        alpha = math.radians(angles[0])
        beta = math.radians(angles[1])
        gamma = math.radians(angles[2])
        
        dof = torch.tensor([math.log(vol), log_a_b, log_c_b, self.theta_to_z(alpha), self.theta_to_z(beta), self.theta_to_z(gamma)])
        return dof

    def wrap_struct(self, s):
        stru = Structure.from_str(s, fmt="poscar")
        #vol = stru.volume

        sga = SpacegroupAnalyzer(stru, symprec=0.2, angle_tolerance=5.0)
        crystal_system = sga.get_crystal_system()
        conv = sga.get_conventional_standard_structure()

        sys_oh, DoF_mask = self.one_hot_encoder(crystal_system)
        debug = False 

        dof = self.get_dof(conv)
        return sys_oh, dof, DoF_mask

    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        g = self.h5[self.keys[idx]]
        coeff = torch.from_numpy(g["coeff"][...])[:, :self.chg_dim, :self.chg_dim, :self.chg_dim]

        sys_oh, dof, DoF_mask = self.wrap_struct(g.attrs["poscar"].decode() if isinstance(g.attrs["poscar"], bytes) else g.attrs["poscar"])

        prop  = torch.tensor([
            g.attrs["energy_above_hull"],
            g.attrs["band_gap"],
            g.attrs["total_magnetization"],
        ])
        return coeff, sys_oh, dof, DoF_mask, prop

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

class TransformedCoeffDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, transform1, transform2):   # transform: CoeffTransform
        self.base_ds  = base_ds
        self.tran1       = transform1
        self.tran2       = transform2

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        coeff, sys_oh, dof, DoF_mask, prop = self.base_ds[idx]
        coeff_tx = self.tran1.forward(coeff)
        dof_tx = self.tran2.forward(dof) 

        return coeff_tx, sys_oh, dof_tx, DoF_mask, prop

def infiniteloop(dataloader):
    while True:
        for data in iter(dataloader):
            coeff = data[0]
            sys_oh = data[1]
            dof = data[2]
            DoF_mask = data[3]
            prop = data[4]
            yield coeff, sys_oh, dof, DoF_mask, prop


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def pad_coeff(coeff_truncated, grid):
    chg_dim = coeff_truncated.shape[0]
    grid = grid
    for i in range(3):
        if grid[i] <= chg_dim:
            grid[i] = chg_dim+1
    coeff = torch.zeros(tuple(grid), device=coeff_truncated.device)
    coeff[:coeff_truncated.shape[0], :coeff_truncated.shape[1], :coeff_truncated.shape[2]] = coeff_truncated
    return coeff

def from_sys_to_mask(sys_rep):
    if type(sys_rep) == torch.Tensor:
        idx = torch.where(sys_rep)[0].item()
        if idx == 0:
            mask = torch.tensor([1,0,0,0,0,0])
        elif idx == 1:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 2:
            mask = torch.tensor([1,1,1,0,0,0])
        elif idx == 3:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 4:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 5:
            mask = torch.tensor([1,1,1,0,0,1])
        elif idx == 6:
            mask = torch.tensor([1,1,1,1,1,1])

    elif type(sys_rep) == np.ndarray:
        idx = np.where(sys_rep)[0].item()
        if idx == 0:
            mask = torch.tensor([1,0,0,0,0,0])
        elif idx == 1:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 2:
            mask = torch.tensor([1,1,1,0,0,0])
        elif idx == 3:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 4:
            mask = torch.tensor([1,0,1,0,0,0])
        elif idx == 5:
            mask = torch.tensor([1,1,1,0,0,1])
        elif idx == 6:
            mask = torch.tensor([1,1,1,1,1,1])
    elif type(sys_rep) == str:
        if sys_rep == "cubic":
            mask = torch.tensor([1,0,0,0,0,0])
        elif sys_rep == "tetragonal":
            mask = torch.tensor([1,0,1,0,0,0])
        elif sys_rep == "orthorhombic":
            mask = torch.tensor([1,1,1,0,0,0])
        elif sys_rep == "hexagonal":
            mask = torch.tensor([1,0,1,0,0,0])
        elif sys_rep == "trigonal":
            mask = torch.tensor([1,0,1,0,0,0])
        elif sys_rep == "monoclinic":
            mask = torch.tensor([1,1,1,0,0,1])
        elif sys_rep == "triclinic":
            mask = torch.tensor([1,1,1,1,1,1])
    else:
        print("unidentified type of sys_rep")
        return None
    return mask

class CoupledDrift(nn.Module):
    def __init__(self, base_model, sys_oh, chg_dim=16):
        super().__init__()
        self.base   = base_model
        self.chg_dim = chg_dim
        self.coeff_sz = 2 * chg_dim**3
        self.dof_sz  = 6
        self.state_sz  = self.coeff_sz + self.dof_sz

        self.sys_oh = sys_oh
        self.dof_mask = torch.stack(list(map(from_sys_to_mask, sys_oh)))


    def forward(self, t, y, args=None):
        B = y.size(0)
        coeff_flat = y[:, :self.coeff_sz]                     # (B, C*D^3)
        latt_flat  = y[:, self.coeff_sz:]                     # (B, 9)

        xt_c = coeff_flat.view(B, 2, self.chg_dim, self.chg_dim, self.chg_dim)
        xt_L = latt_flat.view(B, 6)

        dc_dt, dL_dt = self.base(t, xt_c, self.sys_oh, xt_L, self.dof_mask)
        dy_dt = torch.cat([dc_dt.flatten(1), dL_dt.flatten(1)], dim=1)
        return dy_dt

class CoupledSDE(nn.Module):
    noise_type, sde_type = "diagonal", "ito"

    def __init__(self, drift, score, sys_oh, sigma=0.1, chg_dim=16):
        super().__init__()
        self.drift, self.score, self.sigma = drift, score, sigma
        self.chg_dim = chg_dim
        self.coeff_sz = 2 * chg_dim**3
        self.dof_sz  = 6
        self.state_sz  = self.coeff_sz + self.dof_sz
        self.sys_oh = sys_oh
        self.dof_mask = torch.stack(list(map(from_sys_to_mask, sys_oh)))


    def f(self, t, y_flat, args=None):
        B = y_flat.size(0)
        c_flat = y_flat[:, :self.coeff_sz]        # (B, coeff_sz)
        L_flat = y_flat[:, self.coeff_sz:]        # (B, 9)
        xt_c   = c_flat.view(B, 2, self.chg_dim, self.chg_dim, self.chg_dim)
        xt_L   = L_flat.view(B, 6)
        dc_dt, dL_dt = self.drift(t, xt_c, self.sys_oh, xt_L, self.dof_mask)
        sc_c,  sc_L  = self.score(t, xt_c, self.sys_oh, xt_L, self.dof_mask)

        dy_dt = torch.cat([(dc_dt + sc_c).flatten(1),       # (B, coeff_sz)
            (dL_dt + sc_L).flatten(1)], dim=1)
        return dy_dt

    def g(self, t, y):
        return torch.full_like(y, self.sigma)

def theta_to_z(theta_rad, eps: float = 1e-4):
    """
    Map theta in [0, pi] (radians) to z in (-inf, +inf) via:
      u = 2*theta/pi - 1  in [-1, 1]
      z = atanh(u)
    """
    t = torch.as_tensor(theta_rad, dtype=torch.get_default_dtype(), device=getattr(theta_rad, 'device', None))
    u = 2.0 * t / math.pi - 1.0
    u = u.clamp(min=-1.0 + eps, max=1.0 - eps)
    # atanh(u) = 0.5 * log((1+u)/(1-u))
    z = 0.5 * torch.log((1 + u) / (1 - u))
    return z if isinstance(theta_rad, torch.Tensor) else z.item()

def z_to_theta(z):
    """Local inverse: z -> theta (radians), z = atanh(u), u=tanh(z)."""
    zt = torch.as_tensor(z, dtype=torch.get_default_dtype(), device=getattr(z, "device", None))
    u = torch.tanh(zt)
    return math.pi * (u + 1.0) / 2.0  # in radians

def _system_mask_and_fixed(lat_sys: torch.Tensor, dtype=None, device=None):
    """
    返回 (mask, fixed) 向量（长度=6）。
    mask=1 表示保留输入值；mask=0 表示用 fixed 覆盖。
    角度使用 z 表示：90° -> 0；120° -> 0.5*ln(2) ≈ 0.34657359
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 预计算 z90, z120
    z90  = theta_to_z(torch.tensor(math.pi/2, dtype=dtype, device=device)).to(dtype)
    z120 = 0.34657359027997264  #theta_to_z(torch.tensor(2*math.pi/3, dtype=dtype, device=device)).to(dtype)

    idx = torch.where(lat_sys)[0].item()
    if idx == 0:
        # logV free; a=b=c; α=β=γ=90°
        mask  = torch.tensor([1,0,0,0,0,0], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, z90], dtype=dtype, device=device)

    elif idx == 1:
        # logV, log(c/b) free; a=b; α=β=γ=90°
        mask  = torch.tensor([1,0,1,0,0,0], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, z90], dtype=dtype, device=device)

    elif idx == 2:
        # logV, log(a/b), log(c/b) free; α=β=γ=90°
        mask  = torch.tensor([1,1,1,0,0,0], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, z90], dtype=dtype, device=device)

    elif idx == 3:
        # a=b; α=β=90°, γ=120°; log(c/b) free
        mask  = torch.tensor([1,0,1,0,0,0], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, z120], dtype=dtype, device=device)

    elif idx == 4:
        # 采用 hex setting（若要 rhombohedral，请改为 a=b=c & α=β=γ ≠ 90°）
        mask  = torch.tensor([1,0,1,0,0,0], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, z120], dtype=dtype, device=device)

    elif idx == 5:
        # α=β=90°, γ free
        mask  = torch.tensor([1,1,1,0,0,1], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, z90, z90, 0], dtype=dtype, device=device)

    elif idx == 6:
        # all free
        mask  = torch.tensor([1,1,1,1,1,1], dtype=dtype, device=device)
        fixed = torch.tensor([0,0,0, 0, 0, 0], dtype=dtype, device=device)

    else:
        raise ValueError(f"Unknown lattice system: {lat_sys}")

    return mask, fixed

def enforce_system_on_dof(dof, lat_sys: torch.Tensor):
    """
    dof: (..., 6)  with [logV, log(a/b), log(c/b), zα, zβ, zγ]
    返回修正后的 dof
    """
    x = torch.as_tensor(dof, dtype=torch.get_default_dtype(), device=getattr(dof, 'device', None))
    mask, fixed = _system_mask_and_fixed(lat_sys, dtype=x.dtype, device=x.device)
    while mask.ndim < x.ndim:
        mask  = mask.unsqueeze(0)
        fixed = fixed.unsqueeze(0)
    return x * mask + fixed * (1.0 - mask)

def cell_from_dof_z(dof, lat_sys: torch.Tensor, eps: float = 1e-12):
    """
    输入:
      dof: (..., 6) = [logV, log(a/b), log(c/b), zα, zβ, zγ]
      lat_sys: 晶系字符串（cubic/tetragonal/orthorhombic/hexagonal/trigonal/monoclinic/triclinic）

    输出:
      cell: (..., 3, 3) 列向量分别为 a⃗, b⃗, c⃗
    """
    x = enforce_system_on_dof(dof, lat_sys)
    logV, log_ab, log_cb, z_alpha, z_beta, z_gamma = torch.unbind(x, dim=-1)

    V    = torch.exp(logV)           # volume
    r_ab = torch.exp(log_ab)         # a / b
    r_cb = torch.exp(log_cb)         # c / b

    alpha = z_to_theta(z_alpha)      # radians
    beta  = z_to_theta(z_beta)
    gamma = z_to_theta(z_gamma)

    idx = torch.where(lat_sys)[0].item()

    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sg = torch.sin(gamma).abs().clamp_min(eps)  # 数值稳定

    vol_factor_sq = (1 + 2*ca*cb*cg - ca*ca - cb*cb - cg*cg).clamp_min(eps)
    f = torch.sqrt(vol_factor_sq)

    b = torch.pow(V / (r_ab * r_cb * f), 1.0/3.0)
    a = r_ab * b
    c = r_cb * b

    ax = a
    ay = torch.zeros(())
    az = torch.zeros(())

    bx = b * cg
    by = b * sg
    bz = torch.zeros(())

    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = (c*c) - (cx*cx) - (cy*cy)
    cz = torch.sqrt(torch.clamp(cz_sq, min=0.0))

    cell = torch.stack([
        torch.stack([ax, ay, az]),
        torch.stack([bx, by, bz]),
        torch.stack([cx, cy, cz]),
    ])
    return cell

def write_chgcar(savedir, step, coeff, dof, sys_oh, idx):

    from coeff_transform_paramless import ParamlessTransform, DoFTransform

    tx1 = ParamlessTransform(method="log1p", stat_file="paramless_stats.npz")
    coeff_phy = tx1.inverse(coeff)
    tx2 = DoFTransform(stat_file="dof_stats.npz")
    dof_phy = tx2.inverse(dof)

    cell = cell_from_dof_z(dof_phy, sys_oh)
    # Compute mesh grid from cell
    len_cell = torch.tensor([torch.linalg.norm(v) for v in cell])
    grid = [int(k.item()) for k in torch.floor(len_cell / 0.065)]
    
    if max(grid) > 2000:
        return

    # Padding coeff to grid size
    total_coeff = pad_coeff(coeff_phy[0] + coeff_phy[1], grid)
    diff_coeff = pad_coeff(coeff_phy[0] - coeff_phy[1], grid)

    # IDCT transformation to real space charge density
    total_chg = idctn(total_coeff.detach().cpu().numpy(), norm='ortho')
    diff_chg = idctn(diff_coeff.detach().cpu().numpy(), norm='ortho')

    recon_data = {"total": total_chg, "diff": diff_chg}
    # Construct Structure with given coordinates
    poscar_str = """An artifact chgcar
  1.0
  {:12.6f} {:12.6f} {:12.6f}
  {:12.6f} {:12.6f} {:12.6f}
  {:12.6f} {:12.6f} {:12.6f}
H
1
Direct
  0.500000  0.500000  0.500000
""".format(
        cell[0,0], cell[0,1], cell[0,2],
        cell[1,0], cell[1,1], cell[1,2],
        cell[2,0], cell[2,1], cell[2,2])

    structure = Structure.from_str(poscar_str, fmt="poscar")
    # Output Chgcar for visualization
    chg_recon = Chgcar(structure, recon_data)
    filename = f"{savedir}/recon_chg_step_{step}_{idx}.vasp"
    chg_recon.write_file(filename)
    print(f"Charge density saved to {filename}")

def generate_samples(net_model, savedir, step, chg_dim=16, score_model=None):
    net_model.eval()
    batch = 10
    system_oh = torch.zeros([batch, 7])
    for i in range(batch):
        #system_oh[i, i%7] = 1
        system_oh[i, 0] = 1 # here cubic only

    net_model_ = copy.deepcopy(net_model)
    net_model_ = net_model_.to('cpu')

    device = "cuda"
    coeff_init = torch.randn(batch, 2, chg_dim, chg_dim, chg_dim).to(device)
    dof_init  = torch.randn(batch, 6).to(device)
    y0 = torch.cat([coeff_init.flatten(1), dof_init.flatten(1)], dim=1)

    drift = CoupledDrift(net_model_, system_oh, chg_dim=chg_dim).to(device)
    node  = NeuralODE(drift, solver="euler", sensitivity="adjoint")
    traj = node.trajectory(y0, t_span=torch.linspace(0, 1, 2, device=device))

    coeff_flat, dof_flat = traj[-1][:, :drift.coeff_sz], traj[-1][:, drift.coeff_sz:]
    coeff = coeff_flat.view(batch, 2, chg_dim, chg_dim, chg_dim).to('cpu')
    dof = dof_flat.view(batch, 6).to('cpu')
    for i in range(batch):
        write_chgcar(savedir, step, coeff[i], dof[i], system_oh[i], str(i))

    if score_model:
        y0 = torch.randn(batch, drift.state_sz, device=device)
        sde = CoupledSDE(net_model_, score_model, system_oh, sigma=0.1).to(device)
        with torch.no_grad():
            ts = torch.linspace(0, 1, 2, device=device)
            yT = torchsde.sdeint(sde, y0, ts, dt=0.01)[-1]
            coeff = yT[:, :drift.coeff_sz].view(batch, 2, chg_dim, chg_dim, chg_dim).to('cpu')
            dof  = yT[:, drift.coeff_sz:].view(batch, 6).to('cpu')
        for i in range(batch):
            write_chgcar(savedir, step, coeff[i], dof[i], system_oh[i], str(i)+"_sde")

    net_model.train()
 
