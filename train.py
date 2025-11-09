#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end training script:
H5 (truncated DCT coeffs) -> invert(idctn) -> rho -> SG (rho_to_sg) ->
atoms & Wyckoff (wyckoff_from_rho) -> CNN site embeddings (cnn.py) ->
Graph (node/edge features) -> MPNN -> predict site elements.

This version adds:
- sg_only option in filter_sample_and_dump_debug: pass if SG matches (ignore Wyckoff)
- eval-only mode with per-sample timing (no training)
- validation-only logging: per-mpid SG/Wyckoff (rho & lattice) + match flags, and skip existing on restart
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen|spglib")


import os, sys, re, json, math, h5py, argparse, traceback, importlib.util, time
from typing import List, Tuple, Dict, Any, Optional, Set
import itertools

import numpy as np
from numpy.linalg import inv, norm
from scipy.fftpack import idctn
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset

from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import cnn  # contains SitePatchCNN, site_patch_mean_over_equivalents
import wyckoff_from_rho_final as wy_mod
import rho_to_sg as sg_mod
from rho_to_sg import *



# --- aliases from modules ---
SitePatchCNN = cnn.SitePatchCNN
site_patch_mean_over_equivalents = cnn.site_patch_mean_over_equivalents

DensityField = wy_mod.DensityField
PeakDetector = wy_mod.PeakDetector
atomic_candidates_from_bcps = wy_mod.atomic_candidates_from_bcps
merge_atom_sets = wy_mod.merge_atom_sets
wyckoff_occupancy_from_atoms = wy_mod.wyckoff_occupancy_from_atoms
eq_fracs_from_equiv_groups = wy_mod.eq_fracs_from_equiv_groups

estimate_centering = sg_mod.estimate_centering
select_sg_by_hall = sg_mod.select_sg_by_hall

# ---------------- utils ----------------
def wrap_frac(fr):
    fr = np.asarray(fr, float)
    return fr - np.floor(fr)

def frac_to_cart(frac, lattice):
    return wrap_frac(frac) @ lattice

def minimum_image_cart(v, lattice):
    Linv = inv(lattice)
    f = (v @ Linv)
    f -= np.round(f)
    return f @ lattice

def system_from_sgnum(n: int) -> str:
    if 1 <= n <= 2: return "triclinic"
    if 3 <= n <= 15: return "monoclinic"
    if 16 <= n <= 74: return "orthorhombic"
    if 75 <= n <= 142: return "tetragonal"
    if 143 <= n <= 167: return "trigonal"
    if 168 <= n <= 194: return "hexagonal"
    if 195 <= n <= 230: return "cubic"
    return "triclinic"

def guess_crystal_system(structure: Structure, fallback: str = "orthorhombic") -> str:
    lat: Lattice = structure.lattice
    a,b,c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma
    eps = 1.0
    def eq(x,y,t=1e-2): return abs(x-y) <= t*max(1.0,(abs(x)+abs(y))/2.0)
    if (abs(alpha-90)<eps and abs(beta-90)<eps and abs(gamma-90)<eps):
        if eq(a,b) and eq(b,c): return "cubic"
        if eq(a,b) and not eq(b,c): return "tetragonal"
        return "orthorhombic"
    if abs(alpha-90)<eps and abs(gamma-90)<eps and abs(beta-90)>=eps:
        return "monoclinic"
    if abs(alpha-90)>=eps and abs(beta-90)>=eps and abs(gamma-90)>=eps:
        return "triclinic"
    return fallback

# --------------- invert coeff -> rho ---------------
def idctn_ortho(x: np.ndarray) -> np.ndarray:
    return idctn(x, norm='ortho')

def invert_coeff_to_rho(coeff: np.ndarray,
                        out_shape: Optional[Tuple[int,int,int]] = None,
                        smooth_sigma_vox: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    coeff: (C, L, L, L) truncated DCT coefficients.
    out_shape: if known original grid (Nx,Ny,Nz). If None, use (L,L,L).
    Returns: (rho_total: (Nx,Ny,Nz), info)
    """
    coeff = np.asarray(coeff)
    tgt = coeff.shape[1:]
    C = coeff.shape[0]

    # 两通道判别：up/down（相干） or total/diff（较不相干）
    rec = [idctn_ortho(coeff[i]) for i in range(C)]
    rho_total = rec[0] + rec[1]
    return rho_total.astype(np.float32, copy=False)

# ---------------- dataset ----------------
class H5CoeffDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.keys = sorted([k for k in f.keys() if k.isdigit()], key=lambda s: int(s))
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            g = f[key]
            coeff = np.array(g["coeff"], dtype=np.float32)
            attrs = dict(g.attrs)
            poscar_txt = attrs.get("poscar", "").decode("utf-8") if isinstance(attrs.get("poscar", ""), bytes) else str(attrs.get("poscar", ""))
            mpid = attrs.get("mpid", "")
            if isinstance(mpid, bytes):
                mpid = mpid.decode("utf-8")
        try:
            pos = Poscar.from_str(poscar_txt)
            struct = Structure.from_str(poscar_txt, fmt="poscar")
            sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
            structure = sga.get_conventional_standard_structure()

            sga_lat = SpacegroupAnalyzer(structure, symprec=1e-3)
            sg_lat  = int(sga_lat.get_space_group_number())
            c_system = sga_lat.get_crystal_system()
            ds_lat = sga_lat.get_symmetry_dataset()
        except Exception as e:
            raise RuntimeError(f"Failed to parse POSCAR for {self.h5_path}:{key} ({e})")
        lattice = structure.lattice.matrix.astype(np.float32)

        symm = sga.get_symmetrized_structure()

        wy_labels = symm.wyckoff_symbols            # e.g., ["4a", "8c", ...] per orbit
        orbits = symm.equivalent_sites
        elem_map = {} 
        fr_all = []
        wy_label = []
        equivalent_atoms = []
        for i in range(len(wy_labels)):
            label = re.search(r'([A-Za-z]+)\s*$', wy_labels[i]).group(1)[-1].lower()
            elem = orbits[i][0].species_string
            elem_map[label] = elem
            for j in range(len(orbits[i])):
                fr_all.append(orbits[i][j].frac_coords)
                wy_label.append(label)
                equivalent_atoms.append(i)

        wy_info = { "number": sg_lat,
                   "wyckoffs": wy_label,
                   "equivalent_atoms": equivalent_atoms}

        return {
            "coeff": coeff,           # (2,L,L,L)
            "poscar": poscar_txt,
            "structure": structure,
            "lattice": lattice,       # (3,3) Å
            "mpid": mpid,
            "group_key": key,
            "source": self.h5_path,
            "sg_lat": sg_lat,
            "c_system": c_system,
            "ds_lat": ds_lat,
            "elem_map": elem_map,
            "fr_all": np.array(fr_all),
            "wy_info": wy_info
        }

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[H5CoeffDataset]):
        super().__init__()
        self.datasets = datasets
        self.cum = np.cumsum([len(d) for d in datasets]).tolist()

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx: int):
        if idx < 0: idx += len(self)
        i = int(np.searchsorted(self.cum, idx, side="right"))
        base = 0 if i == 0 else self.cum[i-1]
        return self.datasets[i][idx - base]

# ---------------- rho -> SG ----------------
def predict_spacegroup_from_rho(rho: np.ndarray,
                                lattice: np.ndarray,
                                crystal_system: str,
                                seed: int = 0) -> Dict[str, Any]:
    t_all0 = time.perf_counter()

    # 1) downsample
    t0 = time.perf_counter()
    rho_c = sg_mod.downsample_wrap_to_maxdim(rho, max_dim=64, dtype=np.float32)
    t_downsample = time.perf_counter() - t0

    # 2) estimate_centering
    t0 = time.perf_counter()
    cands, _scores = estimate_centering(rho, crystal_system, n_samples=500, coarse_maxdim=48, seed=seed)
    t_est_center = time.perf_counter() - t0
    if not cands:
        cands = ["P"]

    # 3) 遍历各 centering → select_sg_by_hall
    per_center = []
    best = None
    best_center = None

    t0 = time.perf_counter()
    for cent in cands:
        t1 = time.perf_counter()
        hall_best, _ = select_sg_by_hall(
            rho, rho_c,
            crystal_system=crystal_system,
            n_samples=1000, seed=seed,
            compute_amp_corr=False,
            limit_halls_per_sg=None,
            P_family="identity",
            workers=8,
            bravais_permute_axes=True,
            centering=cent,
            fft_prescreen=True,
            fft_topk=24,
            fft_overlap_thresh=0.20,
        )
        if hall_best == None:
            return None
        t_sel = time.perf_counter() - t1
        per_center.append((cent, t_sel, int(hall_best.hall_number), int(hall_best.sg_number),
                           float(hall_best.avg_rmse), float(hall_best.median_rmse)))

        if best is None or (hall_best.avg_rmse < best.avg_rmse) or \
           (hall_best.avg_rmse == best.avg_rmse and hall_best.median_rmse < best.median_rmse):
            best = hall_best
            best_center = cent
    t_loop = time.perf_counter() - t0

    # 4) 总耗时
    t_total = time.perf_counter() - t_all0
    
    """
    # 5) 打印简单 profile 汇总
    print(f"[PROFILE] predict_spacegroup_from_rho")
    print(f"  total               : {t_total:.4f}s")
    print(f"  ├─ downsample       : {t_downsample:.4f}s")
    print(f"  ├─ estimate_center  : {t_est_center:.4f}s")
    print(f"  └─ loop over centers: {t_loop:.4f}s")
    if per_center:
        print("    per-centering:")
        for cent, tt, hall, sg, avg, med in per_center:
            mark = "← winner" if cent == best_center else ""
            print(f"      - {cent:>2}  {tt:.4f}s  hall={hall:>4}  sg={sg:>3}  avg={avg:.6f}  med={med:.6f} {mark}")
    """
    return {
        "spacegroup_number": int(best.sg_number),
        "spacegroup_symbol": str(best.sg_symbol),
        "hall_number": int(best.hall_number),
        "centering": best_center,
        "avg_rmse": float(best.avg_rmse),
        "median_rmse": float(best.median_rmse),
    }

# ------------- rho -> atoms & wyckoff -------------
from scipy import ndimage as ndi
from typing import Optional, Dict, Any

def atoms_and_wyckoff_from_rho(rho: np.ndarray,
                               lattice: np.ndarray,
                               prefer_sg: Optional[int] = None,
                               smooth_sigma_vox: float = 1.0) -> Dict[str, Any]:
    """
    读取 (rho, lattice)，用 PeakDetector（msblob-only）完成：
      - 明显峰分类（atomic/bond）+ 键峰经 BCP 反推原子
      - 合并去重得到 final_atoms
      - 逐点 Wyckoff（与 final_atoms 一一对应，防止越界）
      - 统计占位（wyckoff_occ），并导出等价分组 eq_groups

    返回字段（与原版对齐）：
      - final_atoms_frac: (N,3) 分数坐标
      - wyckoff_occ: Dict[str,int] 统计
      - spglib_dataset: 与 final_atoms 对齐的 ds_local
      - eq_groups: List[np.ndarray]（每组同等价原子）
      - wyckoff_letters: List[str]（与 final_atoms 一一对应）
      - bond_like_fracs: (M,3) 键峰坐标（来自明显峰分类）
      - bond_like_values: (M,) 这些键峰的峰值
    """
    # ---- 构造 DensityField ----
    class _CHG: pass
    chg = _CHG()
    chg.grid = rho.astype(np.float32, copy=False)
    chg.lattice = lattice.astype(np.float32, copy=False)
    vol = float(np.linalg.det(lattice))
    chg.volume = vol
    chg.dv = vol / float(np.prod(rho.shape))
    chg.total_electrons = float(np.sum(chg.grid) * chg.dv)

    F = DensityField(chg, smooth_sigma_vox=smooth_sigma_vox)
    det = PeakDetector(F)

    # ---- 第一步：分类 + BCP 反推 + 合并 + 逐点 Wyckoff（委托 PeakDetector）----
    sg_input = int(prefer_sg) if (prefer_sg is not None) else 221
    res = det.classify_and_assign_wyckoff(sg=sg_input)
    #print("peak atoms:")
    #for i in res["final_atoms"]:
    #    print(i)
    #print("######################################################################")
    res = wy_mod.refine_missing_via_wyckoff_minima(det.F, res, sg=sg_input)
    #print("merged atoms:")
    #for i in res["final_atoms"]:
    #    print(i)


    # 最终原子（分数坐标）
    final_atoms = np.asarray(res.get("final_atoms", []), dtype=float)
    if final_atoms.ndim != 2:
        final_atoms = np.zeros((0, 3), float)

    # ---- 占位统计（允许内部 standardize_cell；与逐点标注分离）----
    #print(final_atoms)
    wy_occ, wy_info = wyckoff_occupancy_from_atoms(
        lattice, final_atoms, symprec=1e-3,
        prefer_sg=prefer_sg
    )
    #print(wy_occ)
    #print(ds_std)
    #sys.exit()

    # ---- 从“明显峰”里抽取 bond-like 信息（供下游图构建/可视化）----
    peaks_meta = res.get("peaks", []) or []
    bond_like_fracs, bond_like_vals = [], []
    for p in peaks_meta:
        if p.get("class") == "bond":
            f = np.asarray(p.get("frac", [0,0,0]), dtype=float).reshape(-1)
            if f.size == 3:
                bond_like_fracs.append(f)
                bond_like_vals.append(float(p.get("peak_value", 0.0)))

    bond_like_fracs = np.asarray(bond_like_fracs, dtype=float) if bond_like_fracs else np.zeros((0,3), float)
    bond_like_vals  = np.asarray(bond_like_vals,  dtype=float) if bond_like_vals  else np.zeros((0,), float)

    #print("final_atoms_frac", final_atoms)
    #print("wyckoff_occ", wy_occ)
    #print("bond_like_fracs", bond_like_fracs)
    #print("bond_like_values", bond_like_vals)
    #sys.exit()

    return {
        "final_atoms_frac": final_atoms,
        "wyckoff_occ": wy_occ,
        "wyckoff_info": wy_info,
        "bond_like_fracs": bond_like_fracs,
        "bond_like_values": bond_like_vals,
    }


# ------------- SGA lat side (no symmetrized structure) -------------
def _wyckoff_occ_from_structure_lat(struct: Structure, symprec: float = 1e-3) -> Tuple[int, Dict[str, int]]:
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    ds = sga.get_symmetry_dataset()
    sg_lat = int(ds.get("number", sga.get_space_group_number()))
    letters = ds.get("wyckoffs", [])
    occ: Dict[str, int] = {}
    for w in letters:
        if not w: continue
        letter = str(w)[-1].lower()
        occ[letter] = occ.get(letter, 0) + 1
    return sg_lat, occ

def _dict_equal_letters(a: Dict[str, int], b: Dict[str, int]) -> bool:
    letters = set(a.keys()) | set(b.keys())
    for k in letters:
        if int(a.get(k, 0)) != int(b.get(k, 0)):
            return False
    return True

# ---------------- validation-only logging helpers ----------------
def _occ_to_str(occ: Dict[str, int]) -> str:
    if not occ:
        return "-"
    return ",".join(f"{k}:{int(occ[k])}" for k in sorted(occ.keys()))

def _load_logged_mpids(path: str) -> Set[str]:
    mpids: Set[str] = set()
    if not os.path.exists(path):
        return mpids
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                if line.startswith("#") or line.lower().startswith("mpid"):
                    continue
                parts = line.split()
                if parts:
                    mpids.add(parts[0].strip())
    except Exception as e:
        print(f"[VAL-LOG] WARN: failed to read {path}: {e}")
    return mpids

def _append_val_log(path: str, row: Dict[str, str]):
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if new_file: f.write(f"{'mpid':<12} {'match_sg':<9} {'match_wy':<9} {'sg_rho':>6} {'sg_lat':>6} {'wy_rho':<32} {'wy_lat':<32}\n")
        f.write(("{mpid:<12} {match_sg:<9} {match_wy:<9} {sg_rho:>6} {sg_lat:>6} {wy_rho:<32.32} {wy_lat:<32.32}\n").format(**{k:__import__('re').sub(r'\x1b\[[0-9;]*m','',str(v)) for k,v in row.items()}))


def filter_sample_and_dump_debug(
    item: Dict[str, Any],
    rho: np.ndarray,
    sg_pred: Dict[str, Any],
    aw: Dict[str, Any],
    crystal_system_hint: str,
    out_dir: str = "./debug_failed_samples",
    symprec_lat: float = 1e-3,
    verbose: bool = True,
    ds_lat: Optional[dict] = None,
    sg_only: bool = False,   # ★ 新增：只验证 SG
) -> bool:
    """
    Return True if sample passes; otherwise dump CHGCAR/JSON and return False.

    When sg_only=True: only require SG_rho == SG_lat.
    When sg_only=False: require SG equal AND Wyckoff occupancy equal.
    """
    # --- lat side ---
    try:
        if ds_lat is not None:
            sg_lat = int(ds_lat.get("number", -1))
            letters = ds_lat.get("wyckoffs", [])
            occ_lat = {}
            for w in letters:
                if not w: continue
                k = str(w)[-1].lower()
                occ_lat[k] = occ_lat.get(k, 0) + 1
        else:
            sg_lat, occ_lat = _wyckoff_occ_from_structure_lat(item["structure"], symprec=symprec_lat)
    except Exception as e:
        if verbose:
            print(f"[FILTER] SGA failed for {item['source']}[{item['group_key']}]: {e}")
        sg_lat, occ_lat = -1, {}

    # --- rho side (from upstream) ---
    sg_rho = int(sg_pred.get("spacegroup_number", -1))
    occ_rho: Dict[str, int] = {str(k).lower(): int(v) for k, v in (aw.get("wyckoff_occ", {}) or {}).items()}

    # --- pass criteria ---
    if sg_only:
        pass_cond = (sg_rho == sg_lat)
    else:
        pass_cond = (sg_rho == sg_lat) and _dict_equal_letters(occ_rho, occ_lat)



    # --- dump debug if failed ---
    os.makedirs(out_dir, exist_ok=True)
    mpid = item.get("mpid", "NA")
    key  = item.get("group_key", "NA")
    base = f"{mpid}_{key}_rho_SG{sg_rho}_lat_SG{sg_lat}"
    chg_path = os.path.join(out_dir, base + ".CHGCAR")
    meta_path = os.path.join(out_dir, base + ".json")

    if pass_cond:
        print("[PASS] mpid={mpid}  crystal={cs}  SG_rho/lat={sr}/{sl}  "
            "occ_rho={orho}  occ_lat={olat} ".format(
                mpid=mpid, cs=crystal_system_hint, sr=sg_rho, sl=sg_lat,
                orho=occ_rho, olat=occ_lat))
        return True

    #chg = Chgcar(item["structure"], {"total": rho})
    #chg.write_file(chg_path)
    chg_saved = chg_path
    """
    rep = analyze_chgcar_to_sg(
            chg_path,
            crystal_system="cubic",
            n_samples=2000,
            workers=8)
    save_report_json(rep, meta_path)
    """

    print(
            "[FILTER][REJECT] mpid={mpid}  crystal={cs}  SG_rho/lat={sr}/{sl}  "
            "occ_rho={orho}  occ_lat={olat}  mode={mode} -> saved: {p}".format(
                mpid=mpid, cs=crystal_system_hint, sr=sg_rho, sl=sg_lat,
                orho=occ_rho, olat=occ_lat, mode=("sg_only" if sg_only else "sg+wy"),
                p=chg_saved
            )
        )
    return False

sym2Z  = {
    'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
    'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18, 'K':19, 'Ca':20,
    'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
    'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 'Rb':37, 'Sr':38, 'Y':39, 'Zr':40,
    'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50,
    'Sb':51, 'Te':52, 'I':53, 'Xe':54, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60,
    'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70,
    'Lu':71, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
    'Tl':81, 'Pb':82, 'Bi':83}
# ---------------- graph / cnn / mpnn (unchanged) ----------------
import numpy as np
from typing import Dict, Any, List

def build_graph(lattice: np.ndarray,
                fr_all: np.ndarray,
                nn_buffer_A: float = 0.2,
                edge_mode: str = "or") -> Dict[str, Any]:
    """
    Atom-level graph using minimum-image in fractional space.
    edge_mode: 'or' (默认) / 'and' / 'directed'
    """
    L = np.asarray(lattice, float)
    N = len(fr_all)

    def mic_frac(delta: np.ndarray) -> np.ndarray:
        return delta - np.round(delta)

    # pairwise MIC vectors/distances
    F_i = fr_all[:, None, :]
    F_j = fr_all[None, :, :]
    DF = F_j - F_i
    DF_mic = mic_frac(DF)
    V = DF_mic @ L                     # (N,N,3)
    D = np.linalg.norm(V, axis=2)      # (N,N)
    np.fill_diagonal(D, np.inf)

    # per-node min + buffer
    dmin = np.min(D, axis=1)
    cut = np.nextafter(dmin + float(nn_buffer_A), np.inf)  # robust
    keep = D <= cut[:, None]            # i->j

    # 对称策略
    if edge_mode == "and":
        keep_final = keep & keep.T      # 互为邻居才保留
    elif edge_mode == "or":
        keep_final = keep | keep.T      # 任一方向满足
    else:  # 'directed'
        keep_final = keep               # 保持有向，不再补反向

    # 只取上三角避免成对重复，再统一补反向（除了 directed）
    if edge_mode in ("and", "or"):
        ii, jj = np.where(np.triu(keep_final, k=1))
        Vij = V[ii, jj, :].astype(np.float32)     # (E,3)
        # i->j 和 j->i
        src = np.concatenate([ii, jj]).astype(np.int64)
        dst = np.concatenate([jj, ii]).astype(np.int64)
        eattr = np.concatenate([Vij, -Vij], axis=0)
    else:
        ii, jj = np.where(keep_final)
        Vij = V[ii, jj, :].astype(np.float32)
        src = ii.astype(np.int64); dst = jj.astype(np.int64)
        eattr = Vij

    if src.size == 0:
        edge_index = np.zeros((2,0), np.int64)
        edge_attr  = np.zeros((0,3), np.float32)
    else:
        edge_index = np.stack([src, dst], axis=0)
        edge_attr  = eattr

    return {"edge_index": edge_index, "edge_attr": edge_attr}

def one_hot_wy(letter: str) -> np.ndarray:
    idx = max(0, min(25, (ord(letter.lower()) - ord('a'))))
    v = np.zeros(26, np.float32); v[idx] = 1.0
    return v

def sg_one_hot(spacegroup_number: int, n_sg: int = 230) -> np.ndarray:
    """n_sg-dim one-hot for international space group number (1..230)."""
    v = np.zeros(n_sg, dtype=np.float32)
    if isinstance(spacegroup_number, (int, np.integer)) and 1 <= int(spacegroup_number) <= n_sg:
        v[int(spacegroup_number) - 1] = 1.0
    return v

def build_node_features(rho, fr_all, lattice, wyckoff_info, cnn_model, device):
    wy_letters = list(wyckoff_info.get("wyckoffs"))
    N = len(fr_all)
    sg_num = wyckoff_info.get("number")
    sg_vec = torch.from_numpy(sg_one_hot(int(sg_num))).to(device)  # (230,)

    feats = []
    L = lattice.astype(np.float32)

    for f in fr_all:
        patch = site_patch_mean_over_equivalents(
            rho=rho.astype(np.float32),
            lattice=L,
            eq_fracs=np.asarray([f], dtype=np.float32),
            radius=1.0, out_size=32,
        )                               # (32,32,32) numpy
        x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,32,32,32)
        z = cnn_model(x).squeeze(0)     # (embed_dim,)  ← 不要 detach/不要转 numpy
        feats.append(z)

    Z = torch.stack(feats, dim=0)                              # (N, embed_dim)
    wy_oh = torch.from_numpy(np.stack([one_hot_wy(w) for w in wy_letters])).to(device)  # (N,26)
    sg_oh = sg_vec.unsqueeze(0).repeat(N, 1)                   # (N,230)
    #self.wy_proj = nn.Linear(26, hidden, bias=False)
    #self.sg_proj = nn.Linear(230, hidden, bias=False)
    #self.cnn_proj = nn.Linear(embed_dim, hidden, bias=False)
    #X = self.cnn_proj(Z) + self.wy_proj(wy_oh) + self.sg_proj(sg_oh)
    X = torch.cat([Z, wy_oh, sg_oh], dim=-1)                   # (N, embed_dim+26+230)
    return X

import torch
import torch.nn as nn

class MPNN(nn.Module):
    """
    带 Wyckoff 组内硬约束的 MPNN：
      - 输入特征包含 [..., Wy one-hot(K=26), ...]
      - forward 末尾：基于 Wy 分组（argmax）对 logits 做组均值并回填

    参数：
      in_dim:       节点输入维度
      edge_dim:    （可留作兼容占位）
      hidden:       隐藏维度
      num_layers:   消息传递层数
      num_classes:  类别数
      rbf_k:        RBF 基函数数量（编码距离）
      dropout:      Dropout 概率
      jk_mode:      Jumping Knowledge 聚合方式：'concat' | 'max' | 'last'
      wy_start:     x 中 Wy one-hot 的起始下标（例如 128）
      wy_dim:       Wy one-hot 的维度（默认 26）
      enforce_same_wy: 是否启用组内硬约束
    """
    def __init__(self, in_dim: int, edge_dim: int, hidden: int, num_layers: int, num_classes: int,
                 rbf_k: int = 16, dropout: float = 0.10, jk_mode: str = "concat",
                 wy_start: int = -1, wy_dim: int = 26, enforce_same_wy: bool = True):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(num_layers)
        self.rbf_k = int(rbf_k)
        self.dropout = nn.Dropout(dropout)
        self.jk_mode = jk_mode

        # —— Wy one-hot 在输入中的位置，用于组内共享 —— 
        self.wy_start = int(wy_start)            # 例如 128
        self.wy_dim   = int(wy_dim)              # 26
        self.enforce_same_wy = bool(enforce_same_wy)

        # 节点输入映射
        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
        )

        # 边特征编码（方向 + 距离 + RBF）
        self.edge_enc = nn.Sequential(
            nn.Linear(3 + 1 + self.rbf_k, hidden),  # (ux,uy,uz) + r + rbf
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden),
        )

        # 消息 / 更新 / 归一化
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden*3, hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ) for _ in range(self.layers)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden*2, hidden),
                nn.SiLU(inplace=True),
            ) for _ in range(self.layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(self.layers)])

        # JK 聚合
        self.jk_proj = nn.Linear(hidden*(self.layers+1), hidden)

        # 输出头
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    # ======== 工具函数 ========
    @staticmethod
    def _edge_dir_dist(edge_attr: torch.Tensor) -> tuple:
        """从 edge_attr 取相对位移向量 v，拆成单位方向 u 和距离 r。"""
        if edge_attr.numel() == 0:
            device = edge_attr.device
            return (torch.zeros((0,3), device=device, dtype=edge_attr.dtype),
                    torch.zeros((0,1), device=device, dtype=edge_attr.dtype))
        v = edge_attr[..., :3]
        r = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
        u = v / r
        return u, r

    def _rbf(self, r: torch.Tensor, K: int) -> torch.Tensor:
        """对距离 r 做 RBF 基展开（自适应 r_max，稳定梯度）。"""
        if r.numel() == 0:
            return torch.zeros((0, K), device=r.device, dtype=r.dtype)
        with torch.no_grad():
            r_max = torch.quantile(r.detach().squeeze(-1), q=0.95).clamp(min=1.0, max=8.0)
        centers = torch.linspace(0.0, float(r_max), K, device=r.device, dtype=r.dtype).view(1, K)
        delta = (r_max / max(1, K-1)).clamp(min=1e-3)
        gamma = 1.0 / (2.0 * (0.5*delta)**2)
        return torch.exp(-gamma * (r - centers)**2)

    def _encode_edge(self, edge_attr: torch.Tensor) -> torch.Tensor:
        u, r = self._edge_dir_dist(edge_attr)
        rbf = self._rbf(r, self.rbf_k)
        e = torch.cat([u, r, rbf], dim=-1)
        return self.edge_enc(e)

    # ======== 关键：组内硬约束（版本 A：argmax 分组 → 均值 → 回填） ========
    def _enforce_same_wy_logits(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        假设 x 的 [wy_start : wy_start+wy_dim] 是 **硬 one-hot**。
        1) 通过 argmax 得到每个节点的组 id；
        2) 对每个组累加 logits 并计数；
        3) 计算组均值；
        4) 依据组 id 将均值回填到各节点，得到共享 logits。
        """
        if (not self.enforce_same_wy) or (self.wy_start < 0) or (self.wy_dim <= 0):
            return logits

        G = x[:, self.wy_start:self.wy_start + self.wy_dim]  # (N, K)
        if G.numel() == 0:
            return logits

        # 分组 id（0..K-1），硬 one-hot 情况下等价于 argmax
        group_id = G.argmax(dim=-1)  # (N,)
        N, C = logits.size()
        K = self.wy_dim
        device = logits.device
        dtype = logits.dtype

        # 按组累加 logits
        sum_logits = torch.zeros(K, C, device=device, dtype=dtype)
        sum_logits.index_add_(0, group_id, logits)

        # 组计数
        counts = torch.zeros(K, 1, device=device, dtype=dtype)
        counts.index_add_(0, group_id, torch.ones(N, 1, device=device, dtype=dtype))

        # 组均值（避免除零）
        mean_logits = sum_logits / counts.clamp_min(1.0)  # (K, C)

        # 回填：每个节点取其组的均值
        logits_shared = mean_logits.index_select(0, group_id)  # (N, C)
        return logits_shared

    # ======== 前向传播 ========
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 节点初始化
        h = self.node_in(x)
        states = [h]

        # 编码边
        src, dst = edge_index[0], edge_index[1]
        e_enc = self._encode_edge(edge_attr)

        # 消息传递
        for l in range(self.layers):
            if e_enc.size(0) == 0:
                m = torch.zeros_like(h)
            else:
                hi = h[src]
                hj = h[dst]
                m_ij = self.msg_mlps[l](torch.cat([hi, hj, e_enc], dim=-1))  # (E, H)
                m = torch.zeros_like(h)                                     # (N, H)
                m.index_add_(0, dst, m_ij)

            h = self.upd_mlps[l](torch.cat([h, m], dim=-1)) + h
            h = self.norms[l](h)
            h = self.dropout(h)
            states.append(h)

        # JK 聚合
        if self.jk_mode == "concat":
            h_agg = torch.cat(states, dim=-1)
            h_agg = self.jk_proj(h_agg)
        elif self.jk_mode == "max":
            h_agg = torch.stack(states, dim=0).max(dim=0).values
        else:  # 'last'
            h_agg = states[-1]

        # 分类头
        logits = self.out(h_agg)  # (N, C)
        #for i in logits:
        #    print(i)
        #print(np.argmax(logits))
        #print(logits.shape)
        #sys.exit()

        # ★ 在模型内部做 Wyckoff 组内 logits 共享（硬约束）
        logits = self._enforce_same_wy_logits(x, logits)
        return logits

# ---------------- validation helper (minimal) ----------------
@torch.no_grad()
def _compute_validation_loss(test_set: torch.utils.data.Dataset,
                             device: torch.device,
                             cnn_model: nn.Module,
                             mpnn: nn.Module,
                             ce_loss,
                             max_samples: int = -1) -> float:
    """
    验证集上复用训练路径（cubic→搭图→CNN节点特征→MPNN→CE loss），
    并与训练阶段一致地过滤标签空间（Z<=83）。
    """
    # 保留并恢复原训练/评估状态
    mpnn_was_training = mpnn.training
    cnn_was_training = getattr(cnn_model, "training", False)
    mpnn.eval()
    if hasattr(cnn_model, "eval"):
        cnn_model.eval()

    total_loss, count = 0.0, 0
    nsamp = len(test_set) if max_samples < 0 else min(len(test_set), max_samples)

    for i in range(nsamp):
        item = test_set[i]

        # 仅评估 cubic，和主程序一致
        if item.get("c_system", "") != "cubic":
            continue

        coeff = item["coeff"]
        rho = invert_coeff_to_rho(coeff, out_shape=None, smooth_sigma_vox=0.0)

        # —— 与训练路径一致：直接使用数据集里的几何信息（不强依赖在线重建）——
        fr_all  = item.get("fr_all")
        wy_info = item.get("wy_info")
        if fr_all is None or wy_info is None or len(fr_all) == 0:
            continue

        # 搭图（与训练用的默认参数一致）
        G = build_graph(item["lattice"], fr_all, nn_buffer_A=0.2, edge_mode="or")

        # 节点特征（CNN + Wy one-hot + SG one-hot），与训练一致
        X = build_node_features(rho, fr_all, item["lattice"], wy_info, cnn_model, device)

        # 生成标签，严格按训练路径的过滤规则（只保留 Z<=83）
        elem_map   = item.get("elem_map", {})
        wy_letters = wy_info.get("wyckoffs") or []
        # 先保证每个 wy 字母都能在 elem_map 里找到元素
        if any((w not in elem_map) for w in wy_letters):
            continue
        y_symbols = [elem_map[w] for w in wy_letters]

        # 训练代码里：若存在不在 sym2Z 的元素就跳过；这里保持一致
        if any((s not in sym2Z) for s in y_symbols):
            continue

        # to torch
        edge_index = (torch.from_numpy(G["edge_index"]).long().to(device)
                      if G["edge_index"].size > 0 else torch.zeros((2, 0), dtype=torch.long, device=device))
        edge_attr = (torch.from_numpy(G["edge_attr"]).float().to(device)
                     if G["edge_attr"].size > 0 else torch.zeros((0, 3), dtype=torch.float32, device=device))

        # 前向与损失
        logits = mpnn(X, edge_index, edge_attr)  # (N, 83)
        y_idx = torch.tensor([Element(s).Z - 1 for s in y_symbols],
                             dtype=torch.long, device=logits.device)
        # 防御性检查：确保 target 在 [0,82]，否则跳过（理论上已被上面的过滤覆盖）
        if (y_idx.min().item() < 0) or (y_idx.max().item() >= logits.size(-1)):
            continue

        loss = ce_loss(logits, y_idx)
        total_loss += float(loss.detach().cpu().item())
        count += 1

    # 恢复模型状态
    if mpnn_was_training: mpnn.train()
    if hasattr(cnn_model, "train") and cnn_was_training: cnn_model.train()
    return total_loss / max(1, count)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./dataset")
    ap.add_argument("--nfiles", type=int, default=10)
    ap.add_argument("--chg-dim", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.05,
                    help="Train/test split ratio for cached cubic indices.")
    ap.add_argument("--val-interval", type=int, default=5000,
                    help="Run validation every N training steps (samples).")
    ap.add_argument("--val-max-samples", type=int, default=-1,
                    help="Cap validation to this many samples for speed.")
    # ★ 新增：只做一次迭代（不训练），并按样本打印耗时
    ap.add_argument("--eval-only", default=False, action="store_true", help="No training; single pass over data with timing.")
    # ★ 新增：只验证 rho->SG，SG 对即通过ap.add_argument("--sg-only", default="True", action="store_true", help="Debug rho->SG only: pass if SG_rho == SG_lat.")
    ap.add_argument("--sg-only", default=False, action="store_true", help="Debug rho->SG only: pass if SG_rho == SG_lat.")
    ap.add_argument("--val-log", type=str, default="val_log.tsv", help="Validation-only log file (TSV): mpid, sg_rho, wy_rho, sg_lat, wy_lat, match_sg, match_wy.")
    # --- 在 argparse 后面加 ---
    ap.add_argument("--edge-mode", type=str, default="or", choices=["or", "and", "directed"], help="How to symmetrize edges: 'or' keep if i->j or j->i; 'and' keep only mutual; 'directed' keep as-is without adding reverse.")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    h5_paths = [os.path.join(args.data_dir, f"mp_chg_{i:03d}.h5") for i in range(args.nfiles)]
    datasets = [H5CoeffDataset(p) for p in h5_paths if os.path.exists(p)]
    if not datasets:
        raise SystemExit(f"No H5 files found under {args.data_dir}")
    dataset_raw = ConcatDataset(datasets)

    # --- helper: load or build cubic index cache ---
    def _load_or_build_cubic_index(args, dataset_raw, rng):
        idx_cache = os.path.join(args.data_dir, "cubic_idx.txt")
        cubic_idx = []
        if os.path.exists(idx_cache):
            with open(idx_cache, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        j = int(s)
                    except ValueError:
                        continue
                    if 0 <= j < len(dataset_raw):
                        cubic_idx.append(j)
        else:
            # 先枚举索引一次确定 cubic（该步也会触发一次 __getitem__ 解析 POSCAR）
            all_idx = list(range(len(dataset_raw)))
            if args.max_samples >= 0:
                all_idx = all_idx[:min(len(all_idx), args.max_samples)]
            for i in tqdm(all_idx, desc="Scanning for cubic", leave=False, dynamic_ncols=True):
                try:
                    item = dataset_raw[i]
                    c_system = item["c_system"]
                    if c_system == "cubic":
                        cubic_idx.append(i)
                except Exception:
                    continue
            if len(cubic_idx) == 0:
                raise SystemExit("No cubic samples found.")
            os.makedirs(args.data_dir, exist_ok=True)
            with open(idx_cache, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in cubic_idx))
            print(f"[INFO] saved {len(cubic_idx)} cubic indices to {idx_cache}")
        #cubic_idx = cubic_idx[:512]
        rng.shuffle(cubic_idx)
        split = int(round((1.0 - args.test_ratio) * len(cubic_idx)))
        train_idx = cubic_idx[:split]
        test_idx  = cubic_idx[split:]
        print(f"[INFO] cubic samples: {len(cubic_idx)}  -> train: {len(train_idx)}, test: {len(test_idx)}")
        return train_idx, test_idx

    device = torch.device(args.device)
    embed_dim = 128
    cnn_model = SitePatchCNN(embed_dim=embed_dim).to(device).train()
    in_dim = embed_dim + 26 + 230   # CNN + Wy one-hot + SG one-hot
    edge_dim = 3
    hidden = 256
    mpnn_layers = 4 
    mpnn = MPNN(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden, num_layers=mpnn_layers, wy_dim=26, wy_start=embed_dim, num_classes=83).to(device)
    optimizer = torch.optim.Adam(itertools.chain(mpnn.parameters(), cnn_model.parameters()), lr=args.lr)
    #optimizer = torch.optim.AdamW([
    #{'params': mpnn.parameters(),                 'lr': 2e-3},
    #{'params': cnn_model.features.parameters(),   'lr': 1e-3},
    #{'params': cnn_model.proj.parameters(),       'lr': 2e-3},], weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_ckpt_path = "checkpoints/best_model.pt" 

    train_idx, test_idx = _load_or_build_cubic_index(args, dataset_raw, rng)
    train_set = Subset(dataset_raw, train_idx)
    test_set  = Subset(dataset_raw, test_idx)
    active_set = test_set if args.eval_only else train_set
    nsamp = len(active_set) if args.max_samples < 0 else min(len(active_set), args.max_samples)
    ifile = open("log_train", "w")
    print(f"[INFO] Iterating {nsamp} samples on {'test' if args.eval_only else 'train'} split. sg_only={args.sg_only}", file=ifile)

    # 计时统计
    n_proc = 0
    global_step = 0
    acc_invert = acc_sg = acc_atomswy = acc_filter = acc_total = 0.0

    # 只跑一轮；若 eval-only 则不训练
    epochs = 1 if args.eval_only else args.epochs

    # ★ 新增：读取已记录的 mpid 集合（仅 eval-only 有效）
    logged_mpids: Set[str] = _load_logged_mpids(args.val_log) if args.eval_only else set()

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for i in tqdm(range(nsamp), desc=f"Epoch {epoch}", dynamic_ncols=True):
            item = active_set[i]
            mpid = item["mpid"]
            c_system = item["c_system"]
            sg_lat   = item["sg_lat"]
            #if mpid != "mp-1001788":
            #    continue


            # 仅 cubic
            if c_system != "cubic":
                continue
            #print(mpid)

            #if mpid not in ["mp-1001788", "mp-1002183", "mp-1009009", "mp-1014214", "mp-1038814", "mp-1061395",
            #    "mp-1076", "mp-10890", "mp-1094967"]:
            #    continue

            # ★ 新增：eval-only 时，若该 mpid 已存在于日志，直接跳过
            if args.eval_only and mpid and (mpid in logged_mpids):
                print(f"[SKIP-VAL] mpid={mpid} exists in {args.val_log}, skip.")
                continue

            coeff = item["coeff"]

            t0 = time.perf_counter()
            rho = invert_coeff_to_rho(coeff, out_shape=None, smooth_sigma_vox=0.0)
            t1 = time.perf_counter()

            if args.eval_only:
                sg_pred = predict_spacegroup_from_rho(rho, item["lattice"], c_system, seed=args.seed)
                if sg_pred == None:
                    continue
            #print(sg_pred["spacegroup_number"])
            #if sg_pred["spacegroup_number"] != 221:
            #    continue
            t2 = time.perf_counter()

            # atoms & wyckoff
            if args.eval_only:
                try:
                    aw = atoms_and_wyckoff_from_rho(rho, item["lattice"], prefer_sg=sg_pred["spacegroup_number"])
                except Exception as e:
                    print(f"[WARN] atoms/wyckoff failed @ {item['source']}[{item['group_key']}]: {e}")
                    traceback.print_exc()
                    aw = {"wyckoff_occ": {}, "eq_groups": []}
            t3 = time.perf_counter()

            # --- 原有通过/失败过滤与 debug 输出 ---
            '''
            ok = filter_sample_and_dump_debug(
                item=item,
                rho=rho,
                sg_pred=sg_pred,
                aw=aw,
                crystal_system_hint=c_system,
                out_dir="./debug_failed_samples",
                ds_lat=item.get("ds_lat"),
                sg_only=args.sg_only,   # 只验证 SG 时用这个开关
            )
            '''
            t4 = time.perf_counter()

            # ★ 新增：eval-only 记录日志（每个 mpid 仅一行）
            if args.eval_only and mpid:
                # lattice-side wyckoff 计数（优先用 dataset 中的 ds_lat）
                ds_lat = item.get("ds_lat")
                sg_lat = int(ds_lat.get("number", -1))
                letters = ds_lat.get("wyckoffs", [])
                occ_lat = {}
                for w in letters:
                    if not w: continue
                    k = str(w)[-1].lower()
                    occ_lat[k] = occ_lat.get(k, 0) + 1

                # rho-side wyckoff 计数
                occ_rho: Dict[str,int] = {str(k).lower(): int(v) for k, v in (aw.get("wyckoff_occ", {}) or {}).items()}
                # matches
                sg_rho = int(sg_pred.get("spacegroup_number", -1))
                match_sg = (sg_rho == int(sg_lat))
                match_wy = _dict_equal_letters(occ_rho, occ_lat)
                # append
                row = {
                    "mpid": mpid,
                    "sg_rho": str(sg_rho),
                    "wy_rho": _occ_to_str(occ_rho),
                    "sg_lat": str(int(sg_lat)),
                    "wy_lat": _occ_to_str(occ_lat),
                    "match_sg": "True" if match_sg else "False",
                    "match_wy": "True" if match_wy else "False",
                }
                _append_val_log(args.val_log, row)
                logged_mpids.add(mpid)  # 同一进程后续若再遇到同 mpid 也可跳过

                invert_ms  = (t1 - t0) * 1000.0
                sg_ms      = (t2 - t1) * 1000.0
                atomswy_ms = (t3 - t2) * 1000.0
                filter_ms  = (t4 - t3) * 1000.0
                total_ms   = (t4 - t0) * 1000.0
    
                n_proc += 1
                acc_invert  += invert_ms
                acc_sg      += sg_ms
                acc_atomswy += atomswy_ms
                acc_filter  += filter_ms
                acc_total   += total_ms

                print("[TIMING] {i:6s} mpid={mpid} ok={ok}  invert={inv:.1f}ms  sg={sg:.1f}ms  atoms+wy={aw:.1f}ms  filter={ft:.1f}ms  total={tt:.1f}ms".format(
                    i=f"{i}", mpid=item.get("mpid",""), ok=int(ok),
                    inv=invert_ms, sg=sg_ms, aw=atomswy_ms, ft=filter_ms, tt=total_ms
                ))
                print("------------#############################################-----------")

            # eval-only: 不进入训练与图构建
            if args.eval_only:
                continue

            # 下面是原训练路径（目前仅在非 eval-only 时执行）
            #if not ok:
            #    continue

            # --- graph & features ---
            #fr_all = aw.get("final_atoms_frac")
            #wy_info = aw.get("wyckoff_info")
            fr_all = item["fr_all"]
            wy_info = item["wy_info"]
            #print(fr_all)
            #print(wy_info)

            G = build_graph(item["lattice"], fr_all,
                            nn_buffer_A=0.8, edge_mode=args.edge_mode)
            
            X = build_node_features(rho, fr_all, item["lattice"], wy_info, cnn_model, device)
            
            # --- dummy 标签（占位）；你后面可以替换成真实元素/族别等 ---
            elem_map = item["elem_map"]
            y = [elem_map[l] for l in wy_info.get("wyckoffs")]
            skip = False
            for i in range(len(y)):
                if not (y[i] in sym2Z.keys()):
                    skip = True
            if skip:
                continue


            # --- to torch ---
            edge_index = (torch.from_numpy(G["edge_index"]).long().to(device)
                          if G["edge_index"].size > 0 else torch.zeros((2,0), dtype=torch.long, device=device))
            edge_attr  = (torch.from_numpy(G["edge_attr"]).float().to(device)
                          if G["edge_attr"].size > 0 else torch.zeros((0,3), dtype=torch.float32, device=device))
            
            # --- forward / loss / step ---
            logits = mpnn(X, edge_index, edge_attr)
            y_idx = torch.tensor([Element(s).Z - 1 for s in y], dtype=torch.long, device=logits.device)  # (7,)


            loss = F.cross_entropy(logits, y_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().float().cpu().item()

            # --- 每 N 步做一次验证（仅训练模式下；eval-only 不触发） ---
            global_step += 1
            if (not args.eval_only) and (global_step % max(1, int(args.val_interval)) == 0):
                val_loss = _compute_validation_loss(test_set=test_set, device=device, cnn_model=cnn_model, mpnn=mpnn, ce_loss=F.cross_entropy, max_samples=int(args.val_max_samples))
                with open("log_train", "a") as _f:
                    print(f"[VAL@step {global_step}] mean validation loss = {val_loss:.6f}", file=_f)
                
                # 如创新低则保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(best_ckpt_path) or ".", exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": float(val_loss),
                        "mpnn_state_dict": mpnn.state_dict(),
                        "cnn_state_dict": cnn_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    }, best_ckpt_path)
                    msg = f"[BEST] step={global_step}  val_loss={val_loss:.6f}  -> saved to {best_ckpt_path}"
                    with open("log_train", "a") as _f:  # 训练日志
                        print(msg, file=_f)



        if args.eval_only:
            break  # 只做一次迭代
        print(f"[E{epoch}] mean loss={total_loss/max(1,nsamp):.6f}")

    if n_proc > 0:
        print("\n[SUMMARY] processed cubic samples:", n_proc)
        print("  avg invert:   {:.1f} ms".format(acc_invert / n_proc))
        print("  avg rho->SG:  {:.1f} ms".format(acc_sg / n_proc))
        print("  avg atoms&wy: {:.1f} ms".format(acc_atomswy / n_proc))
        print("  avg filter:   {:.1f} ms".format(acc_filter / n_proc))
        print("  avg total:    {:.1f} ms".format(acc_total / n_proc))

    # 仅在训练时保存（eval-only 不保存模型）
    if not args.eval_only:
        cls_map = {"id2elem": []}
        with open("label_space.json", "w", encoding="utf-8") as f:
            json.dump(cls_map, f, indent=2, ensure_ascii=False)
        torch.save(mpnn.state_dict(), "mpnn_site_cls.pt")
        print("✅ Training finished. Saved model to mpnn_site_cls.pt and label_space.json")

if __name__ == "__main__":
    main()

