
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_crystal.py  (NEW pipeline)
==============================
从 sf2m 或 CHGCAR 得到 ρ；
→ detect_peaks_msblob 直接寻峰（原子型峰筛选）；
→ CNN 计算每个 peak 的 atomic embedding；
→ 以 embedding 相似性聚类并分配 dummy labels（A, B, C, ...），并对每个簇做均值作为 shared embedding；
→ 【新】基于 *原子坐标* 做 SG + Wyckoff 分析（spglib + pyXtal 的模板匹配，取自 wyckoff_from_rho_final）
   —— 不再走 “ρ→SG”的老路子；
→ 构图 + 用 *shared embedding* 作为节点 CNN 特征（再拼接 Wyckoff/SG one-hot）；
→ MPNN 元素判别；
→ 输出 POSCAR 与完整 JSON 日志。

依赖：
- 复用你项目中的模块：utils.py, cnn.py, train.py, wyckoff_from_rho_final.py, coeff_transform_paramless.py 等
- 外部：pymatgen、spglib、pyxtal、torch、numpy、scipy

用法示例：
----------
A) 直接从 CHGCAR 推理
    python gen_crystal.py --chgcar path/to/CHGCAR \
        --site-ckpt checkpoints/best_model.pt \
        --out-dir out_new --system cubic

B) 从 sf2m 生成 + 推理
    python gen_crystal.py --sf2m-ckpt checkpoints/sf2m.pt \
        --site-ckpt checkpoints/best_model.pt \
        --out-dir out_new --system cubic

核心参数：
- --sim-thr  : embedding 余弦相似度阈值（簇内最小相似度），默认 0.92
- --z-mad    : detect_peaks_msblob 的 MAD 阈值倍数
- --nms-sep-A: 寻峰 NMS 合并半径（Å）
- --symprec  : spglib 对称分析容差
"""

from __future__ import annotations
import os, sys, json, time, math, argparse, copy, shutil
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from numpy.linalg import norm, inv
from scipy.fftpack import idctn
import subprocess, shlex

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- project-local modules ---
from utils import pad_coeff, cell_from_dof_z
from pymatgen.io.vasp import Chgcar
from pymatgen.io.vasp.inputs  import Poscar
from pymatgen.core import Structure, Lattice, Element

from coeff_transform_paramless import ParamlessTransform, DoFTransform

# site CNN & patch
import cnn
from cnn import SitePatchCNN, sample_site_patch

# peaks, local shape metrics & pyXtal-based wyckoff helper
import wyckoff_from_rho_final as wy_mod
from wyckoff_from_rho_final import DensityField, PeakDetector


# graph & one-hots & MPNN head (重用 train.py 里定义)
from train import build_graph, MPNN, one_hot_wy, sg_one_hot

# optional: sf2m sampler（同你原版脚本）
try:
    from torchcfm.models.unet.unet_v2 import UNetModelWrapper
    from torchdyn.core import NeuralODE
    import torchsde
    HAVE_SF2M = True
except Exception:
    HAVE_SF2M = False


# -------------------- VESTA preview helpers (NEW) --------------------
def _view_with_vesta(vesta_cmd: str, chg_path: str) -> None:
    """Launch VESTA to open CHGCAR and wait for it to close."""
    cmd = [vesta_cmd, chg_path] if " " not in vesta_cmd else shlex.split(vesta_cmd) + [chg_path]
    subprocess.call(cmd)

def _ask_keep_or_regen() -> bool:
    """Return True to keep current CHGCAR, False to regenerate."""
    while True:
        ans = input("do you want to keep the chgcar? otherwise, new chgcar will be regenerated [y/n]: ").strip().lower()
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"):  return False
        print("Please answer with 'y' or 'n'.")


# -------------------- small helpers --------------------
SYS_LIST = ["cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal", "monoclinic", "triclinic"]
def system_one_hot(system: str) -> np.ndarray:
    v = np.zeros((7,), np.float32)
    try:
        v[SYS_LIST.index(system.lower())] = 1.0
    except Exception:
        v[0] = 1.0  # default cubic
    return v

def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out

def write_chgcar_like(total_chg: np.ndarray, lattice: np.ndarray, out_path: str) -> None:
    """用最简结构包装 ρ 并存为 CHGCAR（仅保留 total）。"""
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
    把网络输出的 coeff/dof 逆变换到物理量，并经 pad + IDCT 得到 ρ 与晶胞。
    返回：rho (nx,ny,nz), lattice (3,3)
    """
    tx1 = ParamlessTransform(method="log1p", stat_file="paramless_stats.npz")
    coeff_phy = tx1.inverse(coeff.cpu())
    tx2 = DoFTransform(stat_file="dof_stats.npz")
    dof_phy = tx2.inverse(dof.cpu())

    # 晶胞
    sys_oh_t = torch.from_numpy(sys_oh).float()
    cell = cell_from_dof_z(dof_phy, sys_oh_t)  # torch (3,3); 列向量为 a,b,c
    L = cell.detach().cpu().numpy()

    # 估计网格尺寸：按三边长度 / Δx
    lengths = np.array([norm(L[0]), norm(L[1]), norm(L[2])], float)
    grid = np.floor(lengths / float(grid_spacing_A)).astype(int)
    grid = np.clip(grid, 8, 2000)  # 基本保护

    # 系数拼接与 IDCT
    total_coeff = pad_coeff((coeff_phy[0] + coeff_phy[1]).cpu(), grid.tolist())
    rho = idctn(total_coeff.detach().cpu().numpy(), norm='ortho').astype(np.float32)
    return rho, L

def load_sf2m_models(ckpt_path: str, chg_dim: int, device: torch.device) -> Tuple[nn.Module, Optional[nn.Module]]:
    """最小实现：构建与训练时一致的 U-Net，并加载 ckpt（若含 score_model 也加载）。"""
    if not HAVE_SF2M:
        raise RuntimeError("sf2m not available: torchcfm/torchsde not importable in current env")
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
        score.eval()
    net.eval()
    return net, score

def load_site_models(ckpt_path: str, device: torch.device, embed_dim: int = 128) -> Tuple[nn.Module, nn.Module]:
    """
    CNN + MPNN 推理模型加载。
    - 支持 train.py 写入的 best_model.pt（包含 'cnn_state_dict' 与 'mpnn_state_dict'）
    - 也兼容分别保存的 state_dict（例如仅 mpnn.pt 或 cnn.pt）
    """
    cnn_model = SitePatchCNN(embed_dim=embed_dim).to(device)
    mpnn = MPNN(in_dim=embed_dim + 26 + 230, edge_dim=3, hidden=256, num_layers=4, wy_dim=26, wy_start=embed_dim,  num_classes=83).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and ("cnn_state_dict" in ckpt or "mpnn_state_dict" in ckpt):
        if "cnn_state_dict" in ckpt:
            cnn_model.load_state_dict(ckpt["cnn_state_dict"], strict=False)
        if "mpnn_state_dict" in ckpt:
            mpnn.load_state_dict(ckpt["mpnn_state_dict"], strict=False)
    else:
        # 兼容直接存 state_dict 的场景
        base = os.path.basename(ckpt_path).lower()
        if "cnn" in base:
            cnn_model.load_state_dict(ckpt, strict=False)
        elif "mpnn" in base:
            mpnn.load_state_dict(ckpt, strict=False)
        else:
            # 尝试常见键
            for k in ("state_dict", "model", "model_state_dict"):
                if k in ckpt:
                    try:
                        cnn_model.load_state_dict(ckpt[k], strict=False)
                    except Exception:
                        try:
                            mpnn.load_state_dict(ckpt[k], strict=False)
                        except Exception:
                            pass
    cnn_model.eval(); mpnn.eval()
    return cnn_model, mpnn


# -------------------- embedding → cluster → labels --------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = float(np.linalg.norm(a)); bb = float(np.linalg.norm(b))
    if aa < eps or bb < eps: return 0.0
    return float(np.dot(a, b) / (aa * bb))

@dataclass
class ClusteringResult:
    labels: List[str]                 # length = N sites; 'A','B','C',...
    cluster_ids: List[int]            # numeric ids for convenience
    centers: np.ndarray               # (K, D) mean embeddings
    assign_idx: np.ndarray            # (N,) index in [0..K-1]
    per_cluster_sites: List[List[int]]

def cluster_embeddings_cosine(Z: np.ndarray, sim_thr: float = 0.92) -> ClusteringResult:
    """
    贪心 + 在线更新均值的简单聚类：
      - 顺序遍历；若与最近中心的余弦相似度 >= sim_thr 则并入，否则新建簇
    """
    N, D = Z.shape
    centers, groups = [], []
    assign = np.full(N, -1, int)
    for i in range(N):
        z = Z[i]
        if len(centers) == 0:
            centers.append(z.copy()); groups.append([i]); assign[i] = 0
            continue
        sims = np.array([_cosine_sim(z, c) for c in centers], float)
        j = int(np.argmax(sims))
        if sims[j] >= float(sim_thr):
            groups[j].append(i)
            # 在线更新均值
            c = centers[j]
            n = len(groups[j])
            centers[j] = (c*(n-1) + z) / n
            assign[i] = j
        else:
            centers.append(z.copy()); groups.append([i]); assign[i] = len(centers)-1
    K = len(centers)
    centers = np.stack(centers, axis=0) if K>0 else np.zeros((0,D), np.float32)
    labels = [chr(ord('A') + k) for k in assign]  # A,B,C,...
    return ClusteringResult(labels=labels, cluster_ids=list(range(K)), centers=centers, assign_idx=assign, per_cluster_sites=groups)

def shift_structure_by_nearest_origin(F, lattice=None, tol=1e-12):
    """
    选择“与原点最近的等价像”的原子为锚点，把整组分数坐标整体平移，
    使该原子到原点 (0,0,0)。支持一般（三斜）晶格。

    参数
    ----
    F : (N,3) 分数坐标
    lattice : (3,3) 晶格矩阵；若给出则用笛卡尔距离判断最近原子；否则用分数坐标距离
    tol : 映射回 [0,1) 后把 1.0 保险地归零的容差

    返回
    ----
    G : (N,3) 平移后的分数坐标
    idx : 作为锚点的原子下标
    """
    F = np.asarray(F, float)
    # 最小镜像：映射到 [-0.5, 0.5)
    W = (F + 0.5) % 1.0 - 0.5

    if lattice is not None:
        L = np.asarray(lattice, float)
        d2 = np.sum((W @ L)**2, axis=1)  # 笛卡尔距离^2
    else:
        d2 = np.sum(W**2, axis=1)        # 分数空间距离^2（单位格近似）

    idx = int(np.argmin(d2))             # 最近原子索引
    G = (F - F[idx]) % 1.0               # 整体平移使其到原点
    G[np.isclose(G, 1.0, atol=tol)] = 0.0
    return G, idx


def load_fractional_sites_from_file(path_like: str) -> np.ndarray:
    """Load fractional positions (N,3) from a text file.

    - Supports whitespace-separated values, with optional comment lines starting with '#'
    - Values are wrapped into [0,1)
    """
    p = os.fspath(path_like)
    arr = np.loadtxt(p, dtype=float)
    arr = arr.reshape(-1, 3) % 1.0
    return arr

# -------------------- NEW: pyXtal-based SG + Wy from (fracs, dummy labels) --------------------
def analyze_sg_and_wyckoff_from_sites(lattice: np.ndarray,
                                             fr_all: np.ndarray,
                                             dummy_labels: List[str],
                                             symprec: float = 0.02) -> Dict[str, Any]:
    """
    仅基于 (lattice, fractional coords, types) 的 spglib 结果做 SG & Wyckoff 汇总：
      - 用 spglib.get_symmetry_dataset 得到空间群号、每原子的 Wyckoff 字母、等价分组
      - 以等价分组大小作为该轨道的 multiplicity，生成逐原子的 '4a' 风格标签
      - 汇总每个轨道（等价分组）的分数坐标列表，并统计每个字母的总占位数

    说明：
      - 这里 'types' 仍沿用 dummy_labels（A/B/C…）映射到 1/2/3…；不同 dummy 类型会被视为不同化学种类，
        这会影响 spglib 的等价分组划分（与预期一致：不同元素占同一 Wy 也应拆分成不同轨道）。
      - 结果完全可 JSON 序列化。
    """
    import spglib

    # --- 规范化输入 ---
    L = np.asarray(lattice, float)
    F = np.asarray(fr_all, float).reshape(-1, 3)
    F, anchor_idx = shift_structure_by_nearest_origin(F, lattice=L)


    # dummy label -> type id：A->1, B->2, ...
    lab2id: Dict[str, int] = {}
    types = np.empty(len(F), dtype=int)
    for i, s in enumerate(dummy_labels):
        if s not in lab2id:
            lab2id[s] = len(lab2id) + 1
        types[i] = lab2id[s]

    # --- spglib 对称分析（属性接口，避免弃用告警） ---
    ds = spglib.get_symmetry_dataset((L, F, types), symprec=float(symprec))
    #for i in range(len(F)):
    #    print(F[i], types[i])
    sg_num = int(ds.number)
    wy_letters = list(ds.wyckoffs)                 # 每原子的 Wy 字母
    equiv = list(map(int, ds.equivalent_atoms))    # 等价分组 id（按原子顺序给）
    #print(wy_letters)
    #sys.exit()

    # --- 轨道（等价分组）聚合 & '4a' 标签 ---
    wy_labels_with_mult = [''] * len(F)
    orbits = []  # 记录每个轨道的 {group_id, letter, multiplicity, indices, fracs}
    seen = set()
    for i, g in enumerate(equiv):
        if g in seen:
            continue
        seen.add(g)
        idxs = [j for j, e in enumerate(equiv) if e == g]
        m = len(idxs)
        letter = wy_letters[i]
        for j in idxs:
            wy_labels_with_mult[j] = f"{m}{letter}"
        orbits.append({
            "group_id": g,
            "letter": letter,
            "multiplicity": m,
            "indices": idxs,
            "fracs": F[idxs].tolist(),
        })

    # 每个字母的总占位（把同一字母的多个独立轨道的 multiplicity 求和）
    occupancy_by_letter: Dict[str, int] = {}
    for ob in orbits:
        occupancy_by_letter[ob["letter"]] = occupancy_by_letter.get(ob["letter"], 0) + int(ob["multiplicity"])

    wy_info = {
        "number": sg_num,
        "wyckoffs": wy_letters,                 # 每原子的字母（不带乘数）
        "equivalent_atoms": equiv,              # 每原子的等价分组 id
        "wyckoff_labels_with_mult": wy_labels_with_mult,  # 每原子的 '4a' 风格标签
        "occupancy": occupancy_by_letter,       # {'a':4, 'c':8, ...}
        "orbits": orbits,                       # 轨道级信息（便于可视化/调试）
    }

    # 兼容旧返回结构：eq_fracs_list = 按轨道的分数坐标列表
    eq_fracs_list = [ob["fracs"] for ob in orbits]

    # 给调用方一个可用且可序列化的 spglib 概览（避免把大矩阵塞进 JSON）
    ds_spglib_min = {
        "number": sg_num,
        "international": getattr(ds, "international", None),
        "hall_number": getattr(ds, "hall_number", None),
        "pointgroup": getattr(ds, "pointgroup", None),
    }

    return {"wy_info": wy_info,
            "ds_spglib": ds_spglib_min,
            "eq_fracs_list": eq_fracs_list}


# -------------------- feature builder (use *shared* embeddings) --------------------
def build_node_features_from_embeddings(fr_all: np.ndarray,
                                        wy_info: Dict[str, Any],
                                        embeds: np.ndarray,
                                        device: torch.device) -> torch.Tensor:
    """
    输入:
      - fr_all: (N,3) 分数坐标（只用于长度一致性检查）
      - wy_info: {'number': int, 'wyckoffs': [letter,...]}
      - embeds: (N, D) 预先计算好的节点 embedding（此处传入“簇均值”复制到各节点）
    输出:
      - X: (N, D + 26 + 230) 张量
    """
    fr_all = np.asarray(fr_all, float)
    wy_letters = list(wy_info.get("wyckoffs"))
    sg_num = int(wy_info.get("number"))
    N = len(fr_all)
    assert embeds.shape[0] == N, "embeds row must equal #sites"

    # one-hots
    wy_oh = np.stack([one_hot_wy(w) for w in wy_letters], axis=0).astype(np.float32)   # (N,26)
    sg_oh = np.repeat(sg_one_hot(sg_num)[None, :], N, axis=0).astype(np.float32)       # (N,230)

    # torch
    Z = torch.from_numpy(embeds.astype(np.float32))
    W = torch.from_numpy(wy_oh)
    S = torch.from_numpy(sg_oh)
    X = torch.cat([Z, W, S], dim=-1).to(device)
    return X


# -------------------- main pipeline --------------------
@torch.no_grad()
def run_one(args,
            device: torch.device,
            cnn_model: nn.Module,
            mpnn_model: nn.Module,
            sample_idx: int = 0) -> Dict[str, Any]:

    out_dir = os.path.join(args.out_dir, f"sample_{sample_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    # ---- ρ & lattice ----
    chg_path = os.path.join(out_dir, "CHGCAR")
    if args.chgcar is not None and os.path.isfile(args.chgcar):
        chg_in = Chgcar.from_file(args.chgcar)
        rho = np.asarray(chg_in.data.get("total"), dtype=np.float32)
        lattice = np.asarray(chg_in.structure.lattice.matrix, dtype=float)
        chg_path = os.path.join(out_dir, "CHGCAR")
        try:
            shutil.copy2(args.chgcar, chg_path)
        except Exception:
            chg_in.write_file(chg_path)
        mode = "chgcar"

    
    else:
        # 仅当生成时才进入交互循环
        if not (HAVE_SF2M and args.sf2m_ckpt):
            raise SystemExit("Neither --chgcar provided nor sf2m available. Please pass --chgcar or set up sf2m.")
        while True:
            # 采样一个小流：高斯初值 → ODE 一步（可选 SDE）
            try:
                sys_oh = system_one_hot(args.system)
                sys_oh_batch = torch.from_numpy(sys_oh[None, :]).to(device)
                net, score = load_sf2m_models(args.sf2m_ckpt, chg_dim=args.chg_dim, device=device)
    
                coeff_init = torch.randn(1, 2, args.chg_dim, args.chg_dim, args.chg_dim, device=device)
                dof_init  = torch.randn(1, 6, device=device)
                y0 = torch.cat([coeff_init.flatten(1), dof_init.flatten(1)], dim=1)
    
                from utils import CoupledDrift, CoupledSDE
                drift = CoupledDrift(copy.deepcopy(net).to(device), sys_oh_batch, chg_dim=args.chg_dim).to(device)
                node  = NeuralODE(drift, solver="euler", sensitivity="adjoint")
                traj = node.trajectory(y0, t_span=torch.linspace(0, 1, 2, device=device))
                coeff_flat, dof_flat = traj[-1][:, :drift.coeff_sz], traj[-1][:, drift.coeff_sz:]
                coeff = coeff_flat.view(1, 2, args.chg_dim, args.chg_dim, args.chg_dim).cpu()[0]
                dof   = dof_flat.view(1, 6).cpu()[0]
                mode = "ode"
    
                if (score is not None) and (not args.force_ode):
                    sde = CoupledSDE(copy.deepcopy(net).to(device), copy.deepcopy(score).to(device),
                                     torch.from_numpy(sys_oh[None, :]).to(device), sigma=0.1).to(device)
                    yT = torchsde.sdeint(
                        sde,
                        torch.randn(1, drift.state_sz, device=device),
                        ts=torch.linspace(0, 1, 2, device=device),
                        dt=0.01
                    )[-1]
                    coeff = yT[:, :drift.coeff_sz].view(1, 2, args.chg_dim, args.chg_dim, args.chg_dim).cpu()[0]
                    dof   = yT[:, drift.coeff_sz:].view(1, 6).cpu()[0]
                    mode = "sde"
    
            except Exception as e:
                raise RuntimeError(f"sf2m ODE/SDE sampling failed: {e}")
    
            # → rho & lattice → 写 CHGCAR
            sys_oh = system_one_hot(args.system)
            rho, lattice = coeffdof_to_rho_and_cell(coeff, dof, sys_oh, grid_spacing_A=args.grid_spacing_A)
            write_chgcar_like(rho, lattice, chg_path)
    
            # 仅生成时询问
            #_view_with_vesta(getattr(args, 'vesta_cmd', 'vesta'), chg_path)
            keep = True #_ask_keep_or_regen()
            if keep:
                break
            else:
                try:
                    if os.path.exists(chg_path):
                        os.remove(chg_path)
                except Exception:
                    pass
                print("[INFO] Resampling CHGCAR via sf2m...", flush=True)
                continue

    t1 = time.perf_counter()
    #return {}

    
    # ---- Optional: read fractional sites from file and bypass peak detection ----
    fr_sites = None
    if getattr(args, "fr_sites", None):
        candidates = [args.fr_sites, os.path.join(out_dir, args.fr_sites)]
        for cand in candidates:
            try:
                if cand and os.path.isfile(cand):
                    fr_sites = load_fractional_sites_from_file(cand)
                    print(f"Loaded {len(fr_sites)} fractional sites from {cand}")
                    break
            except Exception as e:
                print(f"[WARN] Failed to load fr_sites from {cand}: {e}")

# ---- ρ → peaks （msblob） + 原子型峰筛选（形状指标） ----
    class _CHG: pass
    chg = _CHG()
    chg.grid = rho.astype(np.float32, copy=False)
    chg.lattice = lattice.astype(np.float32, copy=False)
    vol = float(np.linalg.det(lattice))
    chg.volume = vol
    chg.dv = vol / float(np.prod(rho.shape))
    chg.total_electrons = float(np.sum(chg.grid) * chg.dv)

    F = DensityField(chg)
    det = PeakDetector(F)
    res = det.classify_and_assign_sites()
    if fr_sites is None:
        fr_sites = np.asarray(res["final_atoms"], float).reshape(-1, 3)  # from detector

    print(f"{len(fr_sites)} sites identified")
    t2 = time.perf_counter()

    # ---- CNN embedding（每个 site） ----
    feats = []
    for f in fr_sites:
        patch = sample_site_patch(rho, lattice, f, radius=1.0, out_size=32)   # (32,32,32)
        x = torch.from_numpy(patch[None, None].astype(np.float32)).to(device) # (1,1,32,32,32)
        z = cnn_model(x).squeeze(0).detach().cpu().numpy()
        feats.append(z)
    Z = np.stack(feats, axis=0)  # (N, D)

    # ---- 聚类 & dummy labels ----
    cl = cluster_embeddings_cosine(Z, sim_thr=float(args.sim_thr))
    labels = cl.labels                       # len=N, 诸如 ['A','A','B',...]
    centers = cl.centers                     # (K,D)
    assign = cl.assign_idx                   # (N,)

    # shared embedding: 把 *簇均值* 复制给每个节点
    Z_shared = centers[assign]

    t3 = time.perf_counter()

    # ---- 基于原子型的 SG+Wy（spglib + pyXtal） ----
    ana = analyze_sg_and_wyckoff_from_sites(lattice, fr_sites, labels, symprec=float(args.symprec))
    #print(lattice, fr_sites, labels)
    wy_info = ana["wy_info"]
    eq_fracs_list = ana["eq_fracs_list"]
    print(wy_info)
    #sys.exit()

    t4 = time.perf_counter()

    # ---- 图 + 特征（使用 shared embeddings） ----
    G = build_graph(lattice, fr_sites, nn_buffer_A=0.2, edge_mode="or")
    X = build_node_features_from_embeddings(fr_sites, wy_info, Z_shared, device=device)

    edge_index = (torch.from_numpy(G["edge_index"]).long().to(device)
                  if G["edge_index"].size > 0 else torch.zeros((2,0), dtype=torch.long, device=device))
    edge_attr  = (torch.from_numpy(G["edge_attr"]).float().to(device)
                  if G["edge_attr"].size > 0 else torch.zeros((0,3), dtype=torch.float32, device=device))
    logits = mpnn_model(X, edge_index, edge_attr)  # (N,83)

    # 元素符号（1..83 -> H..Bi）
    idx = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int)  # 0..82
    elem_pred = []
    for k in idx:
        Znum = int(k) + 1
        try:
            elem_pred.append(Element.from_Z(Znum).symbol)
        except Exception:
            elem_pred.append("X")

    for i in range(len(labels)):
        print(labels[i], fr_sites[i], elem_pred[i])
    t5 = time.perf_counter()

    # ---- POSCAR ----
    poscar_path = os.path.join(out_dir, "POSCAR")
    try:
        struct = Structure(Lattice(lattice), species=elem_pred, coords=fr_sites, coords_are_cartesian=False)
        Poscar(struct).write_file(poscar_path)
    except Exception:
        poscar_path = None

    # ---- JSON ----
    meta = {
        "ok": True,
        "mode": ("chgcar" if args.chgcar else "sf2m"),
        "paths": {"CHGCAR": chg_path, "POSCAR": poscar_path},
        "num_sites_used": int(len(fr_sites)),
        "wyckoff_info": wy_info,
        "dummy_labels": labels,
        "cluster_centers_shape": list(centers.shape),
        "elements_pred": elem_pred,
        "timing_ms": {
            "rho_io_or_gen": (t1 - t0) * 1e3,
            "peaks_and_filter": (t2 - t1) * 1e3,
            "cnn_embed_and_cluster": (t3 - t2) * 1e3,
            "sg_wy_from_sites": (t4 - t3) * 1e3,
            "mpnn_elem_pred": (t5 - t4) * 1e3,
            "total": (t5 - t0) * 1e3,
        },
        "params": {
            "sigmas_A": list(args.sigmas_A),
            "z_mad": float(args.z_mad),
            "nms_sep_A": float(args.nms_sep_A),
            "sphere_sphericity_thr": float(args.sphere_sphericity_thr),
            "atom_elong_thr": float(args.atom_elong_thr),
            "sim_thr": float(args.sim_thr),
            "symprec": float(args.symprec),
        },
        "sites": {
            "fracs": fr_sites.tolist(),
            "assign_cluster": assign.tolist(),
            "wyckoff_labels_with_mult": wy_info.get("wyckoff_labels_with_mult"),
        }
    }
    with open(os.path.join(out_dir, "infer_log.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 也保存一个简明的 sg_wy.json
    with open(os.path.join(out_dir, "sg_wy.json"), "w", encoding="utf-8") as f:
        json.dump({
            "wyckoff_info": wy_info,
            "final_atoms_frac": fr_sites.tolist(),
            "elements_pred": elem_pred,
            "dummy_labels": labels,
        }, f, indent=2, ensure_ascii=False)

    return meta


def main():
    p = argparse.ArgumentParser(description="NEW pipeline: peaks->CNN embed->cluster dummy labels->pyXtal SG/Wy->MPNN elements")
    io = p.add_argument_group("I/O")
    io.add_argument("--chgcar", type=str, default=None, help="optional CHGCAR; if omitted will use sf2m to generate rho")
    io.add_argument("--out-dir", type=str, default="out_new2")
    io.add_argument("--site-ckpt", type=str, required=True, help="CNN+MPNN joint ckpt (best_model.pt) or separate state_dict")
    io.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    io.add_argument("--n", type=int, default=1, help="Number of samples generated")
    io.add_argument("--vesta-cmd", type=str, default="vesta", help="command to launch VESTA (e.g., 'vesta' or 'VESTA')")
    io.add_argument("--fr-sites", type=str, default=None, help="Path to fractional positions file (e.g., fr_sites.dat). If set, skip peak detection and use these sites.")


    sf = p.add_argument_group("sf2m (optional)")
    sf.add_argument("--sf2m-ckpt", type=str, default=None)
    sf.add_argument("--chg-dim", type=int, default=16)
    sf.add_argument("--grid-spacing-A", type=float, default=0.065)
    sf.add_argument("--force-ode", action="store_true")
    sf.add_argument("--system", type=str, default="triclinic", choices=SYS_LIST)

    pk = p.add_argument_group("peak detect + filter")
    pk.add_argument("--sigmas-A", dest="sigmas_A", type=float, nargs="+", default=[0.40, 0.55, 0.70, 0.90, 1.20])
    pk.add_argument("--z-mad", type=float, default=3.0)
    pk.add_argument("--nms-sep-A", type=float, default=0.80)
    pk.add_argument("--sphere-sphericity-thr", type=float, default=0.80)
    pk.add_argument("--atom-elong-thr", type=float, default=1.25)

    cl = p.add_argument_group("embedding cluster")
    cl.add_argument("--sim-thr", type=float, default=0.92, help="cosine similarity threshold to merge into an existing cluster")

    sg = p.add_argument_group("symmetry")
    sg.add_argument("--symprec", type=float, default=0.1)

    # misc
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    #torch.manual_seed(int(args.seed))
    #np.random.seed(int(args.seed))
    # choose a base seed: user-provided or time-based
    base_seed = int(time.time_ns() % (2**31 - 1)) if args.seed is None else int(args.seed)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    device = torch.device(args.device)

    # models
    cnn_model, mpnn_model = load_site_models(args.site_ckpt, device=device, embed_dim=128)
    os.makedirs(args.out_dir, exist_ok=True)

    print("sf2m and cnn+mpnn models loaded")
    # 2) 逐样本生成与推理
    manifest = []
    with open(os.path.join(args.out_dir, "run.log"), "a", encoding="utf-8") as log_fh:
        print(f"==== gen_crystal.py run @ {time.strftime('%Y-%m-%d %H:%M:%S')}  "
              #f"ckpt_sf2m={os.path.basename(args.sf2m_ckpt)}  ckpt_site={os.path.basename(args.site_ckpt)} "
              f"n={args.n} sys={args.system} chg_dim={args.chg_dim} device={args.device} ====", file=log_fh)
        for i in range(int(args.n)):
            try:
                per_seed = base_seed + i
                torch.manual_seed(per_seed)
                np.random.seed(per_seed)
                meta = run_one(args, device=device, cnn_model=cnn_model, mpnn_model=mpnn_model, sample_idx=i)
                meta.setdefault("rng", {})["seed"] = per_seed
                manifest.append(meta)
            except Exception as e:
                print(f"[FAIL] i={i:03d}  {e}", file=log_fh, flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Results under: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
