# ===================== Imports =====================
import streamlit as st
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, RegularPolygon
from matplotlib.colors import Normalize
from PIL import Image

from torch_geometric.data.data import DataEdgeAttr

# Allow safe loading of torch_geometric specific objects
torch.serialization.add_safe_globals([DataEdgeAttr])

# ===================== Config =====================
TRIAGE_CACHE  = "triage_results.json"
EXPERT_LABELS = "expert_corrections.json"

st.set_page_config(page_title="Cytogenetics AI Review", layout="wide")

# ===================== Security =====================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Access Restricted")
    pwd = st.text_input("Please enter the password to access this tool:", type="password")
    if st.button("Login"):
        if pwd == "iamwisam":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()  # Stops the rest of the app from running until authenticated

# Rest of the app runs only if authenticated
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

# ===================== Feature Engineering =====================
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
    k = np.ones(win) / win
    out = arr.copy().astype(float)
    for _ in range(passes):
        out = np.convolve(out, k, mode="same")
    return out

def _xspread_profile(pos_xy, n_bins=240):
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

    empty = counts == 0
    if empty.any() and not empty.all():
        xspread[empty] = np.interp(
            np.where(empty)[0], np.where(~empty)[0], xspread[~empty]
        )

    bin_centers = (edges[:-1] + edges[1:]) / 2
    return bin_centers, xspread, y_norm, y_span

def locate_centromere_pinch(bin_centers, xspread, smooth_win=9):
    smoothed = _box_filter(xspread, passes=4, win=smooth_win)
    interior  = (bin_centers >= 0.08) & (bin_centers <= 0.92)
    n         = len(smoothed)
    nbr_half  = 12

    scores = np.full(n, -np.inf)
    for i in range(1, n - 1):
        if not interior[i]:
            continue
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
        interior_spread = smoothed.copy()
        interior_spread[~interior] = np.inf
        best = int(np.argmin(interior_spread))

    return float(np.clip(bin_centers[best], 0.08, 0.92))

def locate_centromere_from_graph(pos_xy, node_intensity, n_bins=720):
    y = pos_xy[:, 1]
    y_norm = (y - y.min()) / (np.ptp(y) + 1e-9)
    intensity_norm = (node_intensity - node_intensity.min()) / (node_intensity.max() - node_intensity.min() + 1e-9)
    
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    intensity_profile = np.zeros(n_bins)
    width_profile = np.zeros(n_bins)
    
    for i in range(n_bins):
        m = (y_norm >= edges[i]) & (y_norm < edges[i + 1])
        if m.sum() >= 2:
            intensity_profile[i] = intensity_norm[m].mean()
            width_profile[i] = np.ptp(pos_xy[m, 0])
    
    width_max = width_profile.max()
    if width_max > 0:
        width_profile = width_profile / width_max
    intensity_profile = (intensity_profile - intensity_profile.min()) / (intensity_profile.max() - intensity_profile.min() + 1e-9)
    
    bin_centers = (edges[:-1] + edges[1:]) / 2
    inverted_width = 1.0 - width_profile
    combined_score = 0.7 * inverted_width + 0.3 * intensity_profile
    
    interior = (bin_centers >= 0.10) & (bin_centers <= 0.90)
    combined_score[~interior] = -np.inf
    
    best_bin = int(np.argmax(combined_score))
    return float(np.clip(bin_centers[best_bin], 0.10, 0.90))

def chromosome_stats(data):
    pos_xy = data.pos.cpu().numpy()
    features = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x
    
    ci_per_node = features[:, -5]         
    arm_indicator = features[:, -4]       
    geodesic_dist = features[:, -3]       
    chrom_width = features[0, -2]         
    chrom_length = features[0, -1]        
    
    ci = float(np.mean(ci_per_node))
    
    y = pos_xy[:, 1]
    y_span = np.ptp(y)
    y_norm = (y - y.min()) / (y_span + 1e-9)
    
    y_sorted_idx = np.argsort(y)
    arm_sorted = arm_indicator[y_sorted_idx]
    
    min_arm_idx = np.argmin(np.abs(arm_sorted))
    centro_y = y_norm[y_sorted_idx[min_arm_idx]]
    
    p_len_px = ci * chrom_length
    q_len_px = (1.0 - ci) * chrom_length
    
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
    h  = y_bot - y_top
    r  = min(r, w / 2, h / 2)
    patch = FancyBboxPatch(
        (x0, y_top), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        clip_on=False,
        **kwargs
    )
    ax.add_patch(patch)
    return patch

def _compute_bands_dynamic(pos_xy, intensity):
    y = pos_xy[:, 1]
    y_norm = (y - y.min()) / (np.ptp(y) + 1e-9)
    n_nodes = len(y_norm)

    n_bins = int(np.clip(n_nodes / 2, 80, 400))
    edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    profile = np.zeros(n_bins)
    counts  = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (y_norm >= edges[i]) & (y_norm < edges[i + 1])
        if m.sum() > 0:
            profile[i] = intensity[m].mean()
            counts[i]  = m.sum()

    empty = counts == 0
    if empty.any() and not empty.all():
        profile[empty] = np.interp(
            np.where(empty)[0], np.where(~empty)[0], profile[~empty]
        )

    smoothed = _box_filter(profile, passes=1, win=3)
    interior = (bin_centers >= 0.02) & (bin_centers <= 0.98)
    thresh   = np.median(smoothed[interior])
    is_dark  = smoothed <= thresh

    min_run = 2
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
    intensity = np.ones(len(pos_xy))
    _, is_dark = _compute_bands_dynamic(pos_xy, intensity)
    if len(is_dark) != len(bin_centers):
        idx = np.round(np.linspace(0, len(is_dark) - 1, len(bin_centers))).astype(int)
        is_dark = is_dark[idx]
    return is_dark

def _binary_bands(bin_centers, xspread, threshold_pct=50):
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
    ax.set_ylim(pos[:, 1].max() + pad, pos[:, 1].min() - pad)
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

    pos_xy    = data.pos.cpu().numpy()
    intensity = data.x[:, 0].cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x[:, 0]
    bin_centers, is_dark = _compute_bands_dynamic(pos_xy, intensity)
    
    DARK  = plt.cm.Greys(0.82)
    LIGHT = plt.cm.Greys(0.06)

    X0    = 0.10
    W     = 0.32
    R_CAP = W / 4
    GAP   = 0.016
    NOTCH = 0.07

    p_top = 0.0
    p_bot = centro_y - GAP / 2
    q_top = centro_y + GAP / 2
    q_bot = 1.0

    def _draw_arm_with_bands(y_top, y_bot):
        h = y_bot - y_top
        r = min(R_CAP, W / 2, h / 2)

        base = FancyBboxPatch(
            (X0, y_top), W, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=LIGHT, edgecolor="none",
            zorder=2, clip_on=False,
        )
        ax.add_patch(base)

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

        outline = FancyBboxPatch(
            (X0, y_top), W, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor="none", edgecolor="#1a1a1a",
            linewidth=1.6, zorder=5, clip_on=False,
        )
        ax.add_patch(outline)

    _draw_arm_with_bands(p_top, p_bot)
    _draw_arm_with_bands(q_top, q_bot)

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

    lx = X0 - 0.05
    ax.text(lx, (p_top + p_bot) / 2, "p",
            ha="right", va="center", fontsize=13,
            fontweight="bold", color="#2c3e50")
    ax.text(lx, (q_top + q_bot) / 2, "q",
            ha="right", va="center", fontsize=13,
            fontweight="bold", color="#2c3e50")

    ax.annotate(
        f"CI = {ci:.3f}",
        xy=(X0 + W, centro_y),
        xytext=(X0 + W + 0.04, centro_y),
        fontsize=9, color="#c0392b", va="center",
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.0),
    )

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

# ===================== UI =================================================
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
    with open(EXPERT_LABELS, "w") as f:
        json.dump(st.session_state.labels, f, indent=2)
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

img_arr = np.rot90(np.array(img), k=3)
ax_img.imshow(img_arr, aspect="equal", cmap="gray" if img_arr.ndim == 2 else None)
ax_img.axis("off")
ax_img.set_title("Original", fontsize=9, pad=4, color="#555")

draw_hex_graph(data, ax_hex)
draw_ideogram(data, ax_ideo)

fig.suptitle("Cytogenetics AI Review", fontsize=12,
             fontweight="bold", color="#2c3e50", y=0.98)

st.pyplot(fig, use_container_width=True)
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
