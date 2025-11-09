#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_crystal.py
==============
One-click pipeline:
  sf2m generation -> coeff/DoF inverse transform -> CHGCAR (ρ) -> ρ→SG → ρ→Wyckoff/atoms
  -> CNN+MPNN site element prediction -> POSCAR
and fully record intermediate artifacts and logs.

Assumptions:
  - You have trained sf2m (net_model/score_model) using train_chg.py and have a checkpoint.
  - You have trained the CNN+MPNN using train.py and saved the weights (best_model.pt or separate state_dict files).
  - This script is for inference/generation only; no training is included.

Output directory structure (--out-dir):
  out/
    sample_000/
      CHGCAR
      POSCAR
      sg_wy.json           # SG prediction and Wyckoff/atoms results
      infer_log.json       # Full pipeline timing and parameters for this sample
    sample_001/...
    run.log                # Aggregated log (text)
    manifest.json          # Summary of all samples

Dependencies: same as the repository (torch, torchsde, torchdyn, pymatgen, scipy, etc.);
This script directly reuses your project's modules/functions:
  - utils.CoupledDrift / CoupledSDE / pad_coeff / cell_from_dof_z
  - train.invert_coeff_to_rho / predict_spacegroup_from_rho / atoms_and_wyckoff_from_rho
  - train.build_graph / build_node_features / MPNN / SitePatchCNN
  - coeff_transform_paramless.ParamlessTransform / DoFTransform
"""
from __future__ import annotations

import os, sys, json, time, copy, math, argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import shutil
from numpy.linalg import norm
from scipy.fftpack import idctn

import torch
import torch.nn as nn

# --- project-local imports ---
from utils import CoupledDrift, CoupledSDE, pad_coeff, cell_from_dof_z
from pymatgen.io.vasp import Chgcar
from pymatgen.io.vasp.inputs  import Poscar
from pymatgen.core import Structure, Lattice, Element

from coeff_transform_paramless import ParamlessTransform, DoFTransform

# ρ→SG / ρ→Wyckoff / graph & features / models
# (Reuse public methods and classes from train.py)
from train import (
    invert_coeff_to_rho,         # coeff -> ρ
    predict_spacegroup_from_rho, # ρ -> SG (number/symbol/centering/...)
    atoms_and_wyckoff_from_rho,  # ρ -> atoms + Wyckoff (bond/ionic pipeline)
    build_graph, build_node_features,
    MPNN, SitePatchCNN
)

# U-Net for sf2m
from torchcfm.models.unet.unet_v2 import UNetModelWrapper
from torchdyn.core import NeuralODE
import torchsde

# -------------------------- helpers --------------------------
SYS_LIST = ["cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal", "monoclinic", "triclinic"]
def system_one_hot(system: str) -> np.ndarray:
    v = np.zeros((7,), np.float32)
    try:
        v[SYS_LIST.index(system.lower())] = 1.0
    except Exception:
        v[0] = 1.0  # default cubic
    return v

def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the 'module.' prefix saved by DDP."""
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out

def load_sf2m_models(ckpt_path: str, chg_dim: int, device: torch.device) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Build the same UNet as in train_chg.py and load the checkpoint (optionally includes a score_model)."""
    net = UNetModelWrapper(
        dim=(2, chg_dim, chg_dim, chg_dim),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1,2,2,2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    score = None
    ckpt = torch.load(ckpt_path, map_location=device)
    if "net_model" in ckpt:
        sd_net = _strip_module_prefix(ckpt["net_model"])
        net.load_state_dict(sd_net, strict=False)
    else:
        # Compatible with checkpoints that directly save a state_dict
        net.load_state_dict(_strip_module_prefix(ckpt), strict=False)
    if "score_model" in ckpt:
        score = UNetModelWrapper(
            dim=(2, chg_dim, chg_dim, chg_dim),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1,2,2,2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(device)
        score.load_state_dict(_strip_module_prefix(ckpt["score_model"]), strict=False)
    net.eval()
    if score is not None:
        score.eval()
    return net, score

def load_site_models(ckpt_path: str, device: torch.device, embed_dim: int = 128) -> Tuple[nn.Module, nn.Module]:
    """
    Load CNN + MPNN inference models.
    - Supports best_model.pt written by train.py (containing 'cnn_state_dict' and 'mpnn_state_dict')
    - Also compatible with separately saved state_dict files (e.g., only mpnn.pt or cnn.pt)
    """
    cnn = SitePatchCNN(embed_dim=embed_dim).to(device)
    mpnn = MPNN(in_dim=embed_dim + 26 + 230, edge_dim=3, hidden=256, num_layers=4, num_classes=83).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "cnn_state_dict" in ckpt or "mpnn_state_dict" in ckpt:
        if "cnn_state_dict" in ckpt:
            cnn.load_state_dict(ckpt["cnn_state_dict"], strict=False)
        if "mpnn_state_dict" in ckpt:
            mpnn.load_state_dict(ckpt["mpnn_state_dict"], strict=False)
    else:
        # Guess by filename keywords to load separately
        if "cnn" in os.path.basename(ckpt_path).lower():
            cnn.load_state_dict(ckpt, strict=False)
        elif "mpnn" in os.path.basename(ckpt_path).lower():
            mpnn.load_state_dict(ckpt, strict=False)
        else:
            # If a combined ckpt is given but the keys are inconsistent, try relaxed key names
            for k in ("state_dict", "model", "model_state_dict"):
                if k in ckpt:
                    try:
                        cnn.load_state_dict(ckpt[k], strict=False)
                    except Exception:
                        try:
                            mpnn.load_state_dict(ckpt[k], strict=False)
                        except Exception:
                            pass
    cnn.eval(); mpnn.eval()
    return cnn, mpnn

def write_chgcar_like(total_chg: np.ndarray, lattice: np.ndarray, out_path: str) -> None:
    """Wrap ρ with a minimal structure and write to CHGCAR."""
    L = np.asarray(lattice, float)
    poscar = f"""Generated CHGCAR holder
  1.0
  {L[0,0]:12.6f} {L[0,1]:12.6f} {L[0,2]:12.6f}
  {L[1,0]:12.6f} {L[1,1]:12.6f} {L[1,2]:12.6f}
  {L[2,0]:12.6f} {L[2,1]:12.6f} {L[2,2]:12.6f}
X
1
Direct
  0.500000  0.500000  0.500000
"""
    struct = Structure.from_str(poscar, fmt="poscar")
    chg = Chgcar(struct, {"total": total_chg})
    chg.write_file(out_path)

def coeffdof_to_rho_and_cell(coeff: torch.Tensor,
                             dof: torch.Tensor,
                             sys_oh: np.ndarray,
                             grid_spacing_A: float = 0.065) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert network outputs coeff/dof back to physical quantities, then pad + IDCT to obtain ρ and the cell.
    Returns: rho (nx,ny,nz), lattice (3,3)
    """
    # Inverse transforms
    tx1 = ParamlessTransform(method="log1p", stat_file="paramless_stats.npz")
    coeff_phy = tx1.inverse(coeff.cpu())
    tx2 = DoFTransform(stat_file="dof_stats.npz")
    dof_phy = tx2.inverse(dof.cpu())

    # Lattice/cell
    sys_oh_t = torch.from_numpy(sys_oh).float()
    cell = cell_from_dof_z(dof_phy, sys_oh_t)  # torch (3,3)
    L = cell.detach().cpu().numpy()

    # Estimate grid size: lengths of three edges / Δx
    lengths = np.array([norm(L[0]), norm(L[1]), norm(L[2])], float)
    grid = np.floor(lengths / float(grid_spacing_A)).astype(int)
    grid = np.clip(grid, 8, 2000)  # basic safeguard

    # Coeff concatenation and IDCT
    total_coeff = pad_coeff((coeff_phy[0] + coeff_phy[1]).cpu(), grid.tolist())
    rho = idctn(total_coeff.detach().cpu().numpy(), norm='ortho').astype(np.float32)
    return rho, L

def take_argmax_Z(logits: torch.Tensor) -> List[str]:
    """(N,83) -> element symbol per site (1..83 -> H..Bi)"""
    idx = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int)  # 0..82
    out = []
    for k in idx:
        Z = int(k) + 1
        try:
            out.append(Element.from_Z(Z).symbol)
        except Exception:
            out.append("X")
    return out

# -------------------------- core pipeline --------------------------
@torch.no_grad()
def generate_one_sample(args,
                        net_model: nn.Module,
                        score_model: Optional[nn.Module],
                        cnn_model: nn.Module,
                        mpnn_model: nn.Module,
                        device: torch.device,
                        sample_idx: int,
                        log_fh) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # --- sample output dir ---
    samp_dir = os.path.join(args.out_dir, f"sample_{sample_idx:03d}")
    os.makedirs(samp_dir, exist_ok=True)
    chg_path = os.path.join(samp_dir, "CHGCAR")

    # Allow two sources: A) read from CHGCAR; B) generate via sf2m
    if args.chgcar is not None:
        # ============ A) Read from CHGCAR ============
        chg_in = Chgcar.from_file(args.chgcar)
        # Copy a copy to the sample directory for traceability
        try:
            shutil.copy2(args.chgcar, chg_path)
        except Exception:
            # If cross-device copy fails, rewrite one
            chg_in.write_file(chg_path)
        lattice = np.asarray(chg_in.structure.lattice.matrix, dtype=float)
        # chg_in.data["total"] is (nx,ny,nz)
        rho = np.asarray(chg_in.data.get("total"), dtype=np.float32)
        mode = "chgcar"
        # Directly proceed to ρ→SG→Wy stage
        sys_oh = system_one_hot(args.system)
    else:
        # ============ B) Generate via sf2m ============
        # --- system one-hot ---
        sys_oh = system_one_hot(args.system)
        sys_oh_batch = torch.from_numpy(sys_oh[None, :]).to(device)

        # --- sf2m: initial state ---
        coeff_init = torch.randn(1, 2, args.chg_dim, args.chg_dim, args.chg_dim, device=device)
        dof_init  = torch.randn(1, 6, device=device)
        y0 = torch.cat([coeff_init.flatten(1), dof_init.flatten(1)], dim=1)

        # --- ODE trajectory (flow) ---
        drift = CoupledDrift(copy.deepcopy(net_model).to(device), sys_oh_batch, chg_dim=args.chg_dim).to(device)
        node  = NeuralODE(drift, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(y0, t_span=torch.linspace(0, 1, 2, device=device))
        coeff_flat, dof_flat = traj[-1][:, :drift.coeff_sz], traj[-1][:, drift.coeff_sz:]
        coeff_ode = coeff_flat.view(1, 2, args.chg_dim, args.chg_dim, args.chg_dim).cpu()
        dof_ode   = dof_flat.view(1, 6).cpu()

        # By default use the SDE terminal state (if score_model exists), otherwise use ODE terminal state
        coeff = coeff_ode[0]; dof = dof_ode[0]
        mode = "ode"
        if score_model is not None and (not args.force_ode):
            sde = CoupledSDE(copy.deepcopy(net_model).to(device), copy.deepcopy(score_model).to(device),
                             torch.from_numpy(sys_oh[None, :]).to(device), sigma=0.1).to(device)
            yT = torchsde.sdeint(sde, torch.randn(1, drift.state_sz, device=device),
                                 ts=torch.linspace(0, 1, 2, device=device), dt=0.01)[-1]
            coeff = yT[:, :drift.coeff_sz].view(1, 2, args.chg_dim, args.chg_dim, args.chg_dim).cpu()[0]
            dof   = yT[:, drift.coeff_sz:].view(1, 6).cpu()[0]
            mode = "sde"
        rho, lattice = coeffdof_to_rho_and_cell(coeff, dof, sys_oh, grid_spacing_A=args.grid_spacing_A)
        write_chgcar_like(rho, lattice, chg_path)
        return {}
    t1 = time.perf_counter()
    t2 = time.perf_counter()

    # --- ρ -> SG ---
    sg = predict_spacegroup_from_rho(rho, lattice, args.system, seed=int(args.seed))
    print("sg:", sg)
    t3 = time.perf_counter()

    # --- ρ -> Wyckoff/atoms (merge bond + ionic routes with min fix) ---
    aw = atoms_and_wyckoff_from_rho(rho, lattice, prefer_sg=int(sg["spacegroup_number"]))
    #print(aw)
    fr_all = np.asarray(aw.get("final_atoms_frac"))
    wy_info = aw.get("wyckoff_info")
    #print("fr_all:", fr_all)
    print("wy_info", wy_info)
    #sys.exit()
    t4 = time.perf_counter()

    # --- CNN+MPNN: site element prediction ---
    if fr_all is None or fr_all.size == 0 or wy_info is None or len(wy_info.get("wyckoffs", [])) == 0:
        elem_pred = []
        logits_np = np.zeros((0,83), np.float32)
    else:
        G = build_graph(lattice, fr_all, nn_buffer_A=0.2, edge_mode="or")
        X = build_node_features(rho, fr_all, lattice, wy_info, cnn_model, device)
        edge_index = (torch.from_numpy(G["edge_index"]).long().to(device)
                      if G["edge_index"].size > 0 else torch.zeros((2,0), dtype=torch.long, device=device))
        edge_attr  = (torch.from_numpy(G["edge_attr"]).float().to(device)
                      if G["edge_attr"].size > 0 else torch.zeros((0,3), dtype=torch.float32, device=device))
        logits = mpnn_model(X, edge_index, edge_attr)  # (N,83)
        elem_pred = take_argmax_Z(logits)
        logits_np = logits.detach().cpu().numpy().astype(np.float32)

    t5 = time.perf_counter()

    # --- Generate POSCAR ---
    if len(elem_pred) == len(fr_all) and len(elem_pred) > 0:
        struct = Structure(Lattice(lattice), species=elem_pred, coords=fr_all, coords_are_cartesian=False)
        poscar_path = os.path.join(samp_dir, "POSCAR")
        Poscar(struct).write_file(poscar_path)
    else:
        poscar_path = None

    # --- per-sample JSON ---
    meta = {
        "sample_index": int(sample_idx),
        "mode": mode,
        "system": args.system,
        "sg_pred": sg,
        "wyckoff_occ": (aw.get("wyckoff_occ") or {}),
        "wyckoff_info": wy_info,
        "num_sites": int(fr_all.shape[0]) if fr_all is not None and fr_all.size else 0,
        "elements_pred": elem_pred,
        "paths": {"CHGCAR": chg_path, "POSCAR": poscar_path},
        "timing_ms": {
            "sf2m": (t1 - t0) * 1e3,
            "invert_to_rho": (t2 - t1) * 1e3,
            "rho_to_sg": (t3 - t2) * 1e3,
            "rho_to_wy": (t4 - t3) * 1e3,
            "elem_pred": (t5 - t4) * 1e3,
            "total": (t5 - t0) * 1e3,
        },
    }
    with open(os.path.join(samp_dir, "infer_log.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(os.path.join(samp_dir, "sg_wy.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sg_pred": sg,
            "wyckoff_occ": (aw.get("wyckoff_occ") or {}),
            "wyckoff_info": wy_info,
            "final_atoms_frac": fr_all.tolist() if fr_all is not None and fr_all.size else [],
            "elements_pred": elem_pred,
        }, f, indent=2, ensure_ascii=False)

    # --- append run.log ---
    wy_str = ",".join([f"{k}:{int(v)}" for k, v in sorted((aw.get("wyckoff_occ") or {}).items())])
    line = (f"[OK] i={sample_idx:03d} mode={mode} sys={args.system:<12} "
            f"SG={sg.get('spacegroup_number','?'):>3} ({sg.get('spacegroup_symbol','?')}) "
            f"N={len(elem_pred):>2}  ρ→SG={meta['timing_ms']['rho_to_sg']:.1f}ms  "
            f"ρ→Wy={meta['timing_ms']['rho_to_wy']:.1f}ms  Elem={meta['timing_ms']['elem_pred']:.1f}ms  "
            f"Wy={wy_str}")
    print(line, file=log_fh, flush=True)

    return meta

# -------------------------- CLI --------------------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end crystal generation: sf2m → ρ → SG/Wy → elements → POSCAR.")
    ap.add_argument("--sf2m-ckpt", required=False, help="Path to weights saved by train_chg.py (includes net_model and optional score_model).")
    ap.add_argument("--site-ckpt", required=True, help="CNN+MPNN checkpoint saved by train.py (best_model.pt, or one of the separate state_dict files).")
    ap.add_argument("--out-dir", type=str, default="./gen_out")
    ap.add_argument("--n", type=int, default=500, help="Number of samples to generate.")
    ap.add_argument("--chg-dim", type=int, default=16, help="Truncated coefficient side length L.")
    ap.add_argument("--system", type=str, default="cubic", choices=SYS_LIST, help="Crystal system one-hot hint.")
    ap.add_argument("--chgcar", type=str, default=None, help="If provided, read ρ and lattice directly from this CHGCAR, skipping the sf2m generation stage.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force-ode", action="store_true", help="Use ODE only even if the checkpoint contains a score_model.")
    ap.add_argument("--grid-spacing-A", type=float, default=0.065, help="Average voxel size (Å) when restoring coefficients to ρ.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device)

    # 1) Load models: when using CHGCAR input, sf2m can be skipped
    if args.chgcar is None:
        if not args.sf2m_ckpt:
            raise ValueError("Missing --sf2m-ckpt (no --chgcar provided).")
        net_model, score_model = load_sf2m_models(args.sf2m_ckpt, args.chg_dim, device)
        print("sf2m models loaded")
    else:
        net_model, score_model = None, None
        print(f"using CHGCAR from: {args.chgcar}")
    cnn_model, mpnn_model = load_site_models(args.site_ckpt, device)

    print("sf2m and cnn+mpnn models loaded")
    # 2) Generate and infer sample-by-sample
    manifest = []
    with open(os.path.join(args.out_dir, "run.log"), "a", encoding="utf-8") as log_fh:
        print(f"==== gen_crystal.py run @ {time.strftime('%Y-%m-%d %H:%M:%S')}  "
              #f"ckpt_sf2m={os.path.basename(args.sf2m_ckpt)}  ckpt_site={os.path.basename(args.site_ckpt)} "
              f"n={args.n} sys={args.system} chg_dim={args.chg_dim} device={args.device} ====", file=log_fh)
        for i in range(int(args.n)):
            try:
                meta = generate_one_sample(args, net_model, score_model, cnn_model, mpnn_model, device, i, log_fh)
                manifest.append(meta)
            except Exception as e:
                print(f"[FAIL] i={i:03d}  {e}", file=log_fh, flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Results under: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()

