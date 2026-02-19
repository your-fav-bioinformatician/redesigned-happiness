# ===================== Imports =====================
import streamlit as st
import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, RegularPolygon
from matplotlib.colors import Normalize
from matplotlib.path import Path
import matplotlib.patches as mpatches
from PIL import Image

from torch_geometric.nn import GATv2Conv, BatchNorm, AttentionalAggregation, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr

torch.serialization.add_safe_globals([DataEdgeAttr])

# ===================== Config =====================
TARGETS_DIR  = "/home/sara/Chr51/testing/notebooks/datasets/targets"
ORPHANS_DIR  = "/home/sara/Chr51/testing/notebooks/datasets/orphans"
PNG_BASE_DIR = "/home/sara/Chr51/testing/data/segments"
MODEL_PATH   = "/home/sara/Chr51/models/gat.pth"

TRIAGE_CACHE  = "triage_results.json"
EXPERT_LABELS = "expert_corrections.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS, HIDDEN, HEADS, ATTN_DROPOUT = 6, 256, 8, 0.2

st.set_page_config(page_title="Cytogenetics AI Review", layout="wide")
st.title("Cytogenetics AI Review")

# ===================== Helpers =====================
def format_chr(c):
    return "X" if c == 23 else "Y" if c == 24 else str(c)

def parse_label(s):
    s = str(s).strip().upper()
    if s == "X": return 23
    if s == "Y": return 24
    if s.isdigit() and 1 <= int(s) <= 24: return int(s)
    return None

# ===================== Feature Engineering (unchanged) =====================
def add_banding_features(data):
    y, base = data.pos[:, 1], data.x[:, 0]
    bins  = 50
    edges = torch.linspace(y.min(), y.max() + 1e-6, bins + 1).to(y.device)
    band  = torch.zeros_like(base)
    for i in range(bins):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if m.any():
            band[m] = base[m].mean()
    idx  = torch.argsort(y)
    g    = band[idx]
    grad = torch.zeros_like(g)
    grad[1:-1] = (g[2:] - g[:-2]) / 2
    grad[0]    = g[1] - g[0]
    grad[-1]   = g[-1] - g[-2]
    grad_full       = torch.zeros_like(band)
    grad_full[idx]  = grad
    data.x = torch.cat([data.x, band[:, None], grad_full[:, None]], dim=1)
    return data


# ===================== Geometry — data.pos only ===========================

def _box_filter(arr, passes=3, win=5):
    """Repeated box filter — approximates Gaussian smoothing."""
    k = np.ones(win) / win
    out = arr.copy().astype(float)
    for _ in range(passes):
        out = np.convolve(out, k, mode="same")
    return out


def _xspread_profile(pos_xy, n_bins=240):
    """
    Bin nodes by normalised y and return x-spread (max-min x) per bin.
    Also returns y_norm (per node) and the raw y-span in pixels.
    """
    x  = pos_xy[:, 0]
    y  = pos_xy[:, 1]
    y_span = float(np.ptp(y))
    y_norm = (y - y.min()) / (y_span + 1e-9)

    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    xspread = np.zeros(n_bins)
    counts  = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        m = (y_norm >= edges[i]) & (y_norm < edges[i + 1])
        if m.sum() >= 2:
            xspread[i] = np.ptp(x[m])
            counts[i]  = m.sum()

    # Fill empty bins by linear interpolation
    empty = counts == 0
    if empty.any() and not empty.all():
        xspread[empty] = np.interp(
            np.where(empty)[0], np.where(~empty)[0], xspread[~empty]
        )

    bin_centers = (edges[:-1] + edges[1:]) / 2
    return bin_centers, xspread, y_norm, y_span


def locate_centromere_pinch(bin_centers, xspread, smooth_win=9):
    """
    Pinch-point prediction: find the most geometrically prominent local
    minimum of the chromosome width profile.

    Algorithm
    ---------
    1. Smooth the raw x-spread profile with a box filter to remove
       single-bin artefacts from sparse node coverage.
    2. Find all local minima in the interior [8%, 92%] of the chromosome
       (excluding telomere tips which can also be narrow).
    3. Score each minimum by its *relative prominence*:
           score = 1 - spread[i] / mean_spread_in_neighbourhood
       A genuine centromeric pinch has high prominence; a gentle taper
       at the shoulder or a noisy dip has low prominence.
    4. Return the highest-scoring minimum as the centromere position.
       Fallback to absolute argmin if no local minima are found.
    """
    smoothed = _box_filter(xspread, passes=4, win=smooth_win)

    interior  = (bin_centers >= 0.08) & (bin_centers <= 0.92)
    n         = len(smoothed)
    nbr_half  = 12   # neighbourhood half-width in bins for prominence calc

    scores = np.full(n, -np.inf)
    for i in range(1, n - 1):
        if not interior[i]:
            continue
        # Must be a local minimum (strictly lower than both neighbours)
        if not (smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]):
            continue
        lo = max(0, i - nbr_half)
        hi = min(n, i + nbr_half + 1)
        nbr = np.concatenate([smoothed[lo:i], smoothed[i + 1:hi]])
        if len(nbr) == 0 or nbr.mean() < 1e-9:
            continue
        scores[i] = 1.0 - smoothed[i] / nbr.mean()

    if np.any(np.isfinite(scores) & (scores > -np.inf)):
        best = int(np.argmax(scores))
    else:
        # Fallback: absolute minimum in interior
        interior_spread = smoothed.copy()
        interior_spread[~interior] = np.inf
        best = int(np.argmin(interior_spread))

    return float(np.clip(bin_centers[best], 0.08, 0.92))


def locate_centromere_from_graph(pos_xy, node_intensity, n_bins=720):
    """
    Enhanced centromere localization using both geometric pinch AND
    heterochromatin intensity signature.
    
    Centromeric regions are typically:
    - Geometrically narrower (low x-spread)
    - More heterochromatic (high intensity values)
    
    This hybrid approach weighs the geometric pinch more heavily but
    incorporates intensity to refine the position.
    """
    y = pos_xy[:, 1]
    y_norm = (y - y.min()) / (np.ptp(y) + 1e-9)
    
    # Normalize intensity (heterochromatin signature)
    intensity_norm = (node_intensity - node_intensity.min()) / (node_intensity.max() - node_intensity.min() + 1e-9)
    
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    intensity_profile = np.zeros(n_bins)
    width_profile = np.zeros(n_bins)
    
    for i in range(n_bins):
        m = (y_norm >= edges[i]) & (y_norm < edges[i + 1])
        if m.sum() >= 2:
            intensity_profile[i] = intensity_norm[m].mean()
            width_profile[i] = np.ptp(pos_xy[m, 0])
    
    # Normalize both profiles
    width_max = width_profile.max()
    if width_max > 0:
        width_profile = width_profile / width_max
    intensity_profile = (intensity_profile - intensity_profile.min()) / (intensity_profile.max() - intensity_profile.min() + 1e-9)
    
    # Centromeric regions: HIGH intensity (heterochromatin) + LOW width (pinch)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Inverted width: narrower = higher score
    inverted_width = 1.0 - width_profile
    
    # Combine: heterochromatin indicator + geometric pinch
    # Weight geometric pinch 70%, intensity 30%
    combined_score = 0.7 * inverted_width + 0.3 * intensity_profile
    
    # Find peak in combined score within interior [10%, 90%]
    interior = (bin_centers >= 0.10) & (bin_centers <= 0.90)
    combined_score[~interior] = -np.inf
    
    best_bin = int(np.argmax(combined_score))
    centro_y = float(np.clip(bin_centers[best_bin], 0.10, 0.90))
    
    return centro_y


def chromosome_stats(data):
    """
    Enhanced chromosome statistics using PRE-COMPUTED features from preprocessing.
    
    Uses reverse indexing to extract the last 5 features appended by preprocessing:
    - features[:, -5] = Centromeric Index (CI)
    - features[:, -4] = Arm indicator (hemisphere: above/below centromere)
    - features[:, -3] = Geodesic distance from centromere (mesh-based)
    - features[:, -2] = Chromosome width
    - features[:, -1] = Chromosome length
    
    These are far more robust than recomputing from scratch.
    
    Returns a dict with:
        centro_y    – normalised centromere position [0,1], 0=top
        ci          – centromere index p / (p + q)
        p_len_px    – p-arm pixel length
        q_len_px    – q-arm pixel length
        chr_len_px  – total chromosome pixel height
        pq_ratio    – p / q
        bin_centers – profile x-axis
        xspread     – raw width profile (for banding)
        smoothed    – smoothed width profile
    """
    pos_xy = data.pos.cpu().numpy()
    features = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x
    
    # Extract pre-computed features using reverse indexing (last 5 columns)
    ci_per_node = features[:, -5]         # Global CI
    arm_indicator = features[:, -4]       # Hemisphere: y - cent_y
    geodesic_dist = features[:, -3]       # Mesh geodesic distance
    chrom_width = features[0, -2]         # Global width (same for all nodes)
    chrom_length = features[0, -1]        # Global length (same for all nodes)
    
    # Use the pre-computed global CI
    ci = float(np.mean(ci_per_node))
    
    # Derive centromere position from arm_indicator
    # arm_indicator is [y - cent_y], so nodes with arm_indicator ≈ 0 are at centromere
    y = pos_xy[:, 1]
    y_span = np.ptp(y)
    y_norm = (y - y.min()) / (y_span + 1e-9)
    
    # Sort by y position and find where arm_indicator crosses zero
    y_sorted_idx = np.argsort(y)
    arm_sorted = arm_indicator[y_sorted_idx]
    
    # Centromere is where arm_indicator is closest to zero
    min_arm_idx = np.argmin(np.abs(arm_sorted))
    centro_y = y_norm[y_sorted_idx[min_arm_idx]]
    
    # Calculate arm lengths using pre-computed CI
    p_len_px = ci * chrom_length
    q_len_px = (1.0 - ci) * chrom_length
    
    # Geometric measurements for banding — 180 bins balances resolution vs noise
    bin_centers, xspread, _, chr_len_px = _xspread_profile(pos_xy, n_bins=180)
    smoothed = _box_filter(xspread, passes=3, win=7)

    return dict(
        centro_y   = centro_y,
        ci         = ci,
        p_len_px   = p_len_px,
        q_len_px   = q_len_px,
        chr_len_px = chrom_length,
        pq_ratio   = p_len_px / q_len_px if q_len_px > 0 else float("nan"),
        bin_centers= bin_centers,
        xspread    = xspread,
        smoothed   = smoothed,
    )


# ===================== Drawing primitives =================================

def _rounded_arm(ax, x0, w, y_top, y_bot, r, zorder=2, **kwargs):
    """
    Draw one chromosome arm as a rounded rectangle.
    r is the corner radius capped at half the smaller dimension so the
    telomere end always forms a full semicircle.
    """
    h  = y_bot - y_top
    r  = min(r, w / 2, h / 2)     # cap so it never exceeds the geometry
    # Use clip_on=False so the rounded cap isn't clipped by the axes box
    patch = FancyBboxPatch(
        (x0, y_top), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        clip_on=False,
        **kwargs
    )
    ax.add_patch(patch)
    return patch


def _compute_bands_dynamic(pos_xy, intensity):
    """
    Fully data-driven G-band simulation using the node intensity profile.

    WHY intensity?
    - Geodesic distance from centromere is MONOTONIC (just increases outward)
      so it can only ever produce 2 zones — useless for multi-band display.
    - Node intensity (data.x[:,0]) reflects local chromatin density and
      genuinely oscillates along the chromosome → real alternating bands.

    Algorithm
    ---------
    1. Auto-choose n_bins from node density along the y-axis so each bin
       has ~3–5 nodes on average (good signal, fine enough resolution).
    2. Build the intensity profile; fill empty bins by interpolation.
    3. Estimate the natural "band frequency" from zero-crossings of the
       raw gradient, then set the smoothing window to suppress anything
       finer than ~half the natural band period — adaptive, not hardcoded.
    4. Threshold at the profile median (50th pct) so dark ≈ light area.
    5. Remove runs shorter than ~1/4 of the median band width so tiny
       speckles are gone but real narrow bands survive.

    Returns (bin_centers, is_dark) where is_dark is bool[n_bins].
    """
    y = pos_xy[:, 1]
    y_norm = (y - y.min()) / (np.ptp(y) + 1e-9)
    n_nodes = len(y_norm)

    # ── 1. Auto bin count: aim for ~2 nodes/bin, clamp to [80, 400] ─────────
    n_bins = int(np.clip(n_nodes / 2, 80, 400))
    edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # ── 2. Intensity profile ─────────────────────────────────────────────────
    profile = np.zeros(n_bins)
    counts  = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (y_norm >= edges[i]) & (y_norm < edges[i + 1])
        if m.sum() > 0:
            profile[i] = intensity[m].mean()
            counts[i]  = m.sum()

    # Fill empty bins
    empty = counts == 0
    if empty.any() and not empty.all():
        profile[empty] = np.interp(
            np.where(empty)[0], np.where(~empty)[0], profile[~empty]
        )

    # ── 3. Minimal smoothing — just enough to kill single-bin shot noise ─────
    # passes=1, win=3 preserves fine oscillations in the intensity profile
    smoothed = _box_filter(profile, passes=1, win=3)

    # ── 4. Threshold at median ───────────────────────────────────────────────
    interior = (bin_centers >= 0.02) & (bin_centers <= 0.98)
    thresh   = np.median(smoothed[interior])
    is_dark  = smoothed <= thresh   # lower intensity = denser chromatin = dark

    # ── 5. Morphological clean-up: only remove truly isolated single bins ────
    min_run = 2  # only kill runs of 1 bin
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n_bins:
            j = i
            while j < n_bins and is_dark[j] == is_dark[i]:
                j += 1
            run_len = j - i
            if run_len < min_run and i > 0:
                is_dark[i:j] = is_dark[i - 1]
                changed = True
            i = j

    return bin_centers, is_dark


def _binary_bands_from_geodesic(bin_centers, pos_xy, geodesic_dist, threshold_pct=50):
    """Kept for API compatibility — delegates to _compute_bands_dynamic."""
    # geodesic_dist is ignored (it is monotonic; see _compute_bands_dynamic)
    # intensity falls back to a uniform stub — caller should use _compute_bands_dynamic directly
    intensity = np.ones(len(pos_xy))  # dummy; real call in draw_ideogram
    _, is_dark = _compute_bands_dynamic(pos_xy, intensity)
    # Resize is_dark to match bin_centers length if needed
    if len(is_dark) != len(bin_centers):
        idx = np.round(np.linspace(0, len(is_dark) - 1, len(bin_centers))).astype(int)
        is_dark = is_dark[idx]
    return is_dark


def _binary_bands(bin_centers, xspread, threshold_pct=50):
    """Kept for API compatibility — width-based fallback with adaptive smoothing."""
    n = len(bin_centers)
    raw_grad = np.diff(xspread)
    zero_crossings = np.where(np.diff(np.sign(raw_grad)))[0]
    if len(zero_crossings) >= 2:
        smooth_win = max(3, int(np.median(np.diff(zero_crossings))))
    else:
        smooth_win = max(3, n // 15)
    smoothed = _box_filter(xspread, passes=3, win=smooth_win)
    inv = smoothed.max() - smoothed
    interior = (bin_centers >= 0.02) & (bin_centers <= 0.98)
    thresh   = np.percentile(inv[interior], threshold_pct)
    is_dark  = inv >= thresh

    n_transitions = max(int(np.sum(np.diff(is_dark.astype(int)) != 0)), 2)
    min_run = max(2, n // (n_transitions * 2))
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and is_dark[j] == is_dark[i]:
                j += 1
            if (j - i) < min_run and i > 0:
                is_dark[i:j] = is_dark[i - 1]
                changed = True
            i = j
    return is_dark


# ===================== Hex graph ==========================================

def draw_hex_graph(data, ax):
    pos  = data.pos.cpu().numpy()
    vals = data.x[:, 0].cpu().numpy()
    cmap = plt.cm.inferno
    norm = Normalize(vals.min(), vals.max())
    r    = 0.015
    for (x, y), v in zip(pos, vals):
        ax.add_patch(RegularPolygon(
            (x, y), 6, radius=r, orientation=np.pi / 6,
            facecolor=cmap(norm(v)), lw=0.3
        ))
    pad = r * 2
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].max() + pad, pos[:, 1].min() - pad)  # flipped: upside down
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    ax.set_title("Node graph", fontsize=9, pad=4, color="#555")




# ===================== Ideogram ===========================================

def draw_ideogram(data, ax):
    s      = chromosome_stats(data)

    centro_y    = s["centro_y"]
    ci          = s["ci"]
    p_len_px    = s["p_len_px"]
    q_len_px    = s["q_len_px"]
    chr_len_px  = s["chr_len_px"]
    pq_ratio    = s["pq_ratio"]
    bin_centers = s["bin_centers"]
    xspread     = s["xspread"]
    smoothed    = s["smoothed"]

    # Dynamic banding: use node intensity (data.x[:,0]) which genuinely oscillates
    # along the chromosome — geodesic distance is monotonic and cannot produce
    # multiple alternating bands, so we always use intensity here.
    pos_xy    = data.pos.cpu().numpy()
    intensity = data.x[:, 0].cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x[:, 0]
    bin_centers, is_dark = _compute_bands_dynamic(pos_xy, intensity)
    
    DARK  = plt.cm.Greys(0.82)
    LIGHT = plt.cm.Greys(0.06)

    X0    = 0.10
    W     = 0.32
    R_CAP = W / 4 # Reduced from W/2 for less rounding
    GAP   = 0.016
    NOTCH = 0.07

    p_top = 0.0
    p_bot = centro_y - GAP / 2
    q_top = centro_y + GAP / 2
    q_bot = 1.0

    def _draw_arm_with_bands(y_top, y_bot):
        h = y_bot - y_top
        r = min(R_CAP, W / 2, h / 2)

        # 1. Filled light base (rounded)
        base = FancyBboxPatch(
            (X0, y_top), W, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=LIGHT, edgecolor="none",
            zorder=2, clip_on=False,
        )
        ax.add_patch(base)

        # 2. Dark strips clipped to the rounded base shape
        m  = (bin_centers >= y_top) & (bin_centers <= y_bot)
        bc = bin_centers[m]
        dk = is_dark[m]
        if len(bc):
            edges = np.concatenate([[y_top], (bc[:-1] + bc[1:]) / 2, [y_bot]])
            for e0, e1, dark in zip(edges[:-1], edges[1:], dk):
                if not dark:
                    continue
                strip = plt.Rectangle(
                    (X0, e0), W, max(e1 - e0, 1e-4),
                    facecolor=DARK, edgecolor="none", zorder=3,
                )
                strip.set_clip_path(base)
                ax.add_patch(strip)

        # 3. Rounded outline on top
        outline = FancyBboxPatch(
            (X0, y_top), W, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor="none", edgecolor="#1a1a1a",
            linewidth=1.6, zorder=5, clip_on=False,
        )
        ax.add_patch(outline)

    _draw_arm_with_bands(p_top, p_bot)
    _draw_arm_with_bands(q_top, q_bot)

    # ---- Centromere constriction -----------------------------------------
    cx = X0 + W / 2
    for y_edge, direction in [(p_bot, +1), (q_top, -1)]:
        depth = GAP * 2.5 * direction
        xs = [X0, cx - NOTCH, cx, cx + NOTCH, X0 + W]
        ys = [y_edge, y_edge, y_edge + depth, y_edge, y_edge]
        ax.fill(xs, ys, color="white", zorder=6)
        ax.plot(xs, ys, color="#1a1a1a", lw=1.0, zorder=6)

    ax.plot([X0 + NOTCH * 0.5, X0 + W - NOTCH * 0.5],
            [centro_y, centro_y],
            color="#c0392b", lw=2.2, zorder=7, solid_capstyle="round")

    # ---- Arm labels -------------------------------------------------------
    lx = X0 - 0.05
    ax.text(lx, (p_top + p_bot) / 2, "p",
            ha="right", va="center", fontsize=13,
            fontweight="bold", color="#2c3e50")
    ax.text(lx, (q_top + q_bot) / 2, "q",
            ha="right", va="center", fontsize=13,
            fontweight="bold", color="#2c3e50")

    # ---- Centromere callout -----------------------------------------------
    ax.annotate(
        f"CI = {ci:.3f}",
        xy=(X0 + W, centro_y),
        xytext=(X0 + W + 0.04, centro_y),
        fontsize=9, color="#c0392b", va="center",
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.0),
    )

    # ---- Stats panel ------------------------------------------------------
    sx = X0 + W + 0.13
    rows = [
        ("Centromere Index",  f"{ci:.4f}",            "p / (p + q)"),
        ("Chromosome Length", f"{chr_len_px:.1f} px",  "y-span of nodes"),
        ("p-arm Length",      f"{p_len_px:.1f} px",    f"{ci:.1%} of total"),
        ("q-arm Length",      f"{q_len_px:.1f} px",    f"{1-ci:.1%} of total"),
        ("p / q Ratio",
         f"{pq_ratio:.3f}" if np.isfinite(pq_ratio) else "N/A",
         "p_len / q_len"),
    ]

    ax.text(sx, -0.05, "CHROMOSOME STATS",
            fontsize=9, fontweight="bold", color="#aaaaaa", va="top")

    y_txt = 0.06
    for title, value, note in rows:
        ax.plot([sx, sx + 0.55], [y_txt - 0.006, y_txt - 0.006],
                color="#eeeeee", lw=0.8, zorder=1)
        ax.text(sx, y_txt + 0.004, title,
                fontsize=8.5, color="#999999", va="top")
        ax.text(sx, y_txt + 0.040, value,
                fontsize=15, color="#111111", va="top", fontweight="bold")
        ax.text(sx, y_txt + 0.115, note,
                fontsize=7.5, color="#cccccc", va="top", style="italic")
        y_txt += 0.185

    # ---- Band legend ------------------------------------------------------
    leg_y = 0.96
    ax.text(sx, leg_y - 0.032, "BAND KEY",
            fontsize=8, fontweight="bold", color="#aaaaaa", va="top")
    for offset, label, fc in [(0.0,  "Heterochromatin (narrow)", DARK),
                               (0.07, "Euchromatin (wide)",       LIGHT)]:
        ax.add_patch(plt.Rectangle(
            (sx, leg_y + offset), 0.045, 0.042,
            facecolor=fc, edgecolor="#666", lw=0.6, zorder=10
        ))
        ax.text(sx + 0.062, leg_y + offset + 0.021, label,
                fontsize=8, color="#555555", va="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.06, -0.06)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_title("Ideogram", fontsize=10, pad=6, color="#555")



# ===================== UI — single fixed 16:9 figure ======================

if not os.path.exists(TRIAGE_CACHE):
    st.info("No triage cache found.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx    = 0
    st.session_state.data   = json.load(open(TRIAGE_CACHE))
    st.session_state.labels = (json.load(open(EXPERT_LABELS))
                                if os.path.exists(EXPERT_LABELS) else {})

def save(label):
    k = st.session_state.data[st.session_state.idx]["id"]
    st.session_state.labels[k] = label
    json.dump(st.session_state.labels, open(EXPERT_LABELS, "w"), indent=2)
    st.session_state.idx += 1

if st.session_state.idx >= len(st.session_state.data):
    st.success("All reviewed.")
    st.stop()

item = st.session_state.data[st.session_state.idx]
st.progress(st.session_state.idx / len(st.session_state.data))

# ---- Load data --------------------------------------------------------
data   = torch.load(item["pt_path"], map_location="cpu", weights_only=False)
img    = Image.open(item["png_path"])

# ---- Single 16:9 figure -----------------------------------------------
FIG_W, FIG_H = 16, 9
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white", dpi=110)

# Three columns: [image | hex graph | ideogram+stats]
# Image and hex get equal space; ideogram gets more for the stats panel
gs = gridspec.GridSpec(
    1, 3,
    figure=fig,
    width_ratios=[3, 2.5, 4],
    left=0.01, right=0.99,
    top=0.93,  bottom=0.03,
    wspace=0.05,
)

ax_img  = fig.add_subplot(gs[0])
ax_hex  = fig.add_subplot(gs[1])
ax_ideo = fig.add_subplot(gs[2])

# Panel 1 – original image
img_arr = np.rot90(np.array(img), k=3)   # 90° clockwise
ax_img.imshow(img_arr, aspect="equal", cmap="gray" if img_arr.ndim == 2 else None)
ax_img.axis("off")
ax_img.set_title("Original", fontsize=9, pad=4, color="#555")

# Panel 2 – hex graph
draw_hex_graph(data, ax_hex)

# Panel 3 – ideogram
draw_ideogram(data, ax_ideo)

# Figure title
fig.suptitle("Cytogenetics AI Review", fontsize=12,
             fontweight="bold", color="#2c3e50", y=0.98)

st.pyplot(fig, width='content')
plt.close(fig)

# ---- Label buttons ----------------------------------------------------
st.divider()
cols = st.columns(len(item["preds"]))
for i, (k, v) in enumerate(item["preds"].items()):
    if cols[i].button(f"Chr {k} ({v}%)"):
        save(parse_label(k))
        st.rerun()

with st.form("manual"):
    v = st.text_input("Manual label")
    if st.form_submit_button("Save"):
        p = parse_label(v)
        if p:
            save(p)
            st.rerun()