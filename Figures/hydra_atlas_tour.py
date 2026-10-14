"""
HYDRA atlas tour — animated cover figure
=========================================

Cycles through all ten HYDRA climate indices. For each index it shows:
  * left:  the index name and a small animated time series of the index
           (long-term 1901–2024 for ocean indices, intermediate-term 1941–2024
           for atmospheric indices), with the short-term period (1981–2024) shaded
  * centre: an Atlantic-centred global map of the DJF regression of SST (ocean)
            and CRU precipitation (land) onto the index, short-term period
  * right: colour bars for SST and precipitation

Colour limits are set per index (95th percentile of |regression|), so every
index's map is fully saturated; the colour bars update with each index.

Outputs (written to OUT_DIR):
  hydra_atlas_tour.gif   looping GIF (for slides / posts)
  hydra_atlas_tour.mp4   H.264 video (recommended for the website)
  hydra_atlas_tour_poster.png   still of the first frame (website fallback)

Requirements: xarray, netCDF4, numpy, matplotlib, cartopy, pillow, ffmpeg on PATH

Usage:
  python hydra_atlas_tour.py
Edit the CONFIG block below to change paths, season, size or timing.
"""

import io
import os
import shutil
import subprocess
import tempfile

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image

# ───────────────────────────── CONFIG ─────────────────────────────
DATA_DIR = "hydra_data"              # folder containing indices/ and teleconnections/
OUT_DIR = "."                        # where the GIF / MP4 / poster are written
COASTLINE_SHP = None                 # path to a Natural Earth coastline .shp, or None
                                     # to let cartopy download it automatically
SEASON = "DJF"                       # DJF, MAM, JJA or SON
CENTRAL_LON = 0                      # 0 = Atlantic-centred; 180 = Pacific-centred
FIG_W, FIG_H, DPI = 32 / 3, 4, 150   # inches → 1600 × 600 px
FPS = 12
REVEAL_FRAMES = 22                   # frames to draw the time series
HOLD_FRAMES = 10                     # frames to hold the finished panel
FADE_FRAMES = 8                      # cross-fade frames to the next index

INDICES = ["ENSO", "PDO", "IOD", "AMO", "TPI", "AAO", "AO", "SAM", "NAO", "NPI"]
OCEAN = {"ENSO", "PDO", "IOD", "AMO", "TPI"}
# period labels as used on the HYDRA website (DJF seasons are dated by their January)
PERIODS = {"LT": ("1901", "2024"), "IT": ("1941", "2024")}

BG = "#0f2a20"          # site dark green
TEXT = "#ffffff"
SUBTLE = "#95d5b2"
MUTED = "#7fa894"
# brown → teal (precipitation / PDSI style, matching the HYDRA figures)
BRBG = LinearSegmentedColormap.from_list(
    "hydra_brbg", ["#543005", "#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5",
                   "#c7eae5", "#5ab4ac", "#01665e", "#003c30"])
SST_CMAP = "RdBu_r"
NICE_SST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
NICE_PPT = [5, 8, 10, 12, 15, 20, 25, 30, 40]
# ──────────────────────────────────────────────────────────────────


def nice_limit(values, choices, pct=95):
    """Smallest 'nice' limit >= the pct-th percentile of |values|."""
    v = np.nanpercentile(np.abs(values), pct)
    for c in choices:
        if c >= v:
            return c
    return choices[-1]


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_index_series():
    """Full-record seasonal series: LT for ocean indices, IT for atmospheric."""
    lt = xr.open_dataset(os.path.join(DATA_DIR, "indices", "SST_INDEX_LT.nc"))
    it = xr.open_dataset(os.path.join(DATA_DIR, "indices", "ATM_INDEX_IT.nc"))
    series = {}
    for k in INDICES:
        da = lt[k] if k in OCEAN else it[k]
        years = da.time.dt.year.values + (da.time.dt.month.values - 1) / 12
        series[k] = (years, da.values.astype(float))
    return series


def render_base(k):
    """Static part of one panel: name, map, gridlines, coastlines, colour bars."""
    ds = xr.open_dataset(os.path.join(DATA_DIR, "teleconnections", f"{k}_SST_PPT_CRU_REG_ST.nc"))
    sst = ds[f"{k}_SST_reg"].sel(season=SEASON)
    ppt = ds[f"{k}_PPT_CRU_reg"].sel(season=SEASON)
    ppt = ppt.where(ppt.lat > -60)             # CRU has no Antarctica
    sst_lim = nice_limit(sst.values, NICE_SST)
    ppt_lim = nice_limit(ppt.values, NICE_PPT)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)

    # map
    ax = fig.add_axes([0.235, 0.04, 0.64, 0.92], projection=ccrs.Robinson(central_longitude=CENTRAL_LON))
    ax.set_facecolor(BG)
    ax.spines["geo"].set_edgecolor("#2d4a3e")
    ms = ax.pcolormesh(ds.lon, ds.lat, sst, cmap=SST_CMAP, vmin=-sst_lim, vmax=sst_lim,
                       transform=ccrs.PlateCarree(), shading="auto")
    mp = ax.pcolormesh(ds.lon, ds.lat, ppt, cmap=BRBG, vmin=-ppt_lim, vmax=ppt_lim,
                       transform=ccrs.PlateCarree(), shading="auto")
    if COASTLINE_SHP:
        import cartopy.io.shapereader as shpreader
        coast = cfeature.ShapelyFeature(shpreader.Reader(COASTLINE_SHP).geometries(),
                                        ccrs.PlateCarree(), facecolor="none")
        ax.add_feature(coast, edgecolor="#1f1f1f", linewidth=0.45)
    else:
        ax.coastlines(resolution="50m", color="#1f1f1f", linewidth=0.45)
    gl = ax.gridlines(color="#ffffff", alpha=0.35, linewidth=0.4, linestyle="--")
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
    ax.set_global()

    # colour bars (right)
    for cax_pos, mappable, label, ticks in [
        ([0.895, 0.56, 0.013, 0.34], ms, f"SST (°C per σ)", [-sst_lim, 0, sst_lim]),
        ([0.895, 0.10, 0.013, 0.34], mp, f"Precip. (% per σ)", [-ppt_lim, 0, ppt_lim]),
    ]:
        cax = fig.add_axes(cax_pos)
        cb = fig.colorbar(mappable, cax=cax, ticks=ticks, extend="both")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=SUBTLE, labelsize=7, length=0)
        cb.set_label(label, color=SUBTLE, fontsize=7.5, labelpad=4)

    # left text
    fig.text(0.03, 0.80, k, color=TEXT, fontsize=30, fontweight="bold", va="center")
    fig.text(0.03, 0.06, f"{SEASON} regression onto {k}, 1981–2024\nHYDRA · HYdrological Drought & Rainfall Atlas",
             color=MUTED, fontsize=6.5, va="bottom")
    return fig_to_image(fig)


def render_series(k, series, frac, size_px):
    """Small time-series panel, revealed up to fraction `frac` of the record."""
    w, h = size_px
    years, vals = series[k]
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.0, 0.22, 1.0, 0.62])
    ax.set_facecolor(BG)
    start = years[0]
    ax.axvspan(1981, years[-1] + 0.25, color="#ffffff", alpha=0.08, lw=0)
    n = max(1, int(round(frac * len(vals))))
    v = np.clip(vals[:n], -3, 3)
    colors = np.where(v >= 0, "#5ab4ac", "#d8b365")
    ax.bar(years[:n], v, width=0.25, color=colors, linewidth=0)
    ax.axhline(0, color=MUTED, lw=0.4)
    ax.set_xlim(start - 0.5, years[-1] + 0.5)
    ax.set_ylim(-3.2, 3.2)
    ax.axis("off")
    period = "LT" if k in OCEAN else "IT"
    first, last = PERIODS[period]
    fig.text(0.0, 0.93, f"{period} {first}–{last}", color=SUBTLE, fontsize=6.5, va="top")
    st_x = (1981 - (start - 0.5)) / ((years[-1] + 0.5) - (start - 0.5))
    fig.text(st_x, 0.93, "ST 1981–2024", color=TEXT, fontsize=6.5, va="top")
    fig.text(0.0, 0.02, first, color=MUTED, fontsize=6, va="bottom")
    fig.text(1.0, 0.02, last, color=MUTED, fontsize=6, va="bottom", ha="right")
    return fig_to_image(fig)


def main():
    series = load_index_series()
    W, H = int(FIG_W * DPI), int(FIG_H * DPI)
    # time-series box, in pixels (left column, under the index name)
    box = (int(0.03 * W), int(0.30 * H), int(0.19 * W), int(0.34 * H))  # x, y(top), w, h

    bases = {k: render_base(k) for k in INDICES}
    frames = []
    for i, k in enumerate(INDICES):
        nxt = INDICES[(i + 1) % len(INDICES)]
        full = None
        for f in range(REVEAL_FRAMES + HOLD_FRAMES):
            frac = min(1.0, (f + 1) / REVEAL_FRAMES)
            img = bases[k].copy()
            img.paste(render_series(k, series, frac, box[2:]), box[:2])
            frames.append(img)
            full = img
        for t in range(1, FADE_FRAMES):
            frames.append(Image.blend(full, bases[nxt], t / FADE_FRAMES))

    os.makedirs(OUT_DIR, exist_ok=True)
    frames[REVEAL_FRAMES + HOLD_FRAMES - 1].save(os.path.join(OUT_DIR, "hydra_atlas_tour_poster.png"))

    tmp = tempfile.mkdtemp()
    try:
        for j, fr in enumerate(frames):
            fr.save(os.path.join(tmp, f"{j:04d}.png"))
        src = os.path.join(tmp, "%04d.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", src,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                        os.path.join(OUT_DIR, "hydra_atlas_tour.mp4")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", src,
                        "-vf", "scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=200[p];"
                               "[b][p]paletteuse=dither=sierra2_4a",
                        "-loop", "0", os.path.join(OUT_DIR, "hydra_atlas_tour.gif")], check=True)
    finally:
        shutil.rmtree(tmp)
    print(f"Wrote {len(frames)} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
