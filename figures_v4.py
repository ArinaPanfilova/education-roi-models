"""Рисунки для итоговой статьи (v4): прогнозы входов модели и IRR. Запускать после model_v3.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT, FIG, DATA = BASE / "out_v3", BASE / "figures_v4", BASE / "data"
FIG.mkdir(exist_ok=True)
R = json.loads((OUT / "results.json").read_text())
S = pd.read_pickle(OUT / "state.pkl")
macro = pd.read_csv(DATA / "macro.csv")
prem = pd.read_csv(DATA / "premium.csv")
ger = pd.read_csv(DATA / "ger.csv")

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "savefig.dpi": 300, "figure.dpi": 150,
})
# различимы и в ч/б печати: цвет + тип линии + маркер
STY = {
    "RUS": dict(color="#1f3b73", ls="-", marker="o", label="Россия"),
    "CHN": dict(color="#b03a2e", ls="--", marker="s", label="Китай"),
    "ZAF": dict(color="#5d6d3a", ls=":", marker="^", label="ЮАР"),
}
W_IN = 4.5   # ширина набора на листе А5 (~11,4 см)
from matplotlib.ticker import FuncFormatter
COMMA = FuncFormatter(lambda v, p: (f"{v:g}").replace(".", ",").replace("-", "−"))


def commas(fig):
    for a in fig.axes:
        a.yaxis.set_major_formatter(COMMA)
H = np.arange(2026, 2031)


def fan(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- Рис. 1: прогнозы входов модели
fig, axes = plt.subplots(1, 3, figsize=(W_IN, 2.0))
for iso, st in STY.items():
    d = macro[macro.iso == iso]
    um, gm = R["ts"][iso]["unemp_model"], R["ts"][iso]["growth_model"]
    ax = axes[0]
    ax.plot(d.year, d.unemp, color=st["color"], ls=st["ls"], lw=0.9, label=st["label"])
    ax.plot(np.r_[2025, H], np.r_[d.unemp.iloc[-1], um["forecast"]], color=st["color"], ls=st["ls"], lw=1.5)
    fan(ax, H, um["lo80"], um["hi80"], st["color"])
    g = 100 * np.log(d.gdppc_ppp).diff()
    ax = axes[1]
    ax.plot(d.year.iloc[1:], g.iloc[1:], color=st["color"], ls=st["ls"], lw=0.9)
    ax.plot(np.r_[2025, H], np.r_[g.iloc[-1], gm["forecast"]], color=st["color"], ls=st["ls"], lw=1.5)
    fan(ax, H, gm["lo80"], gm["hi80"], st["color"])
    p = prem[prem.iso == iso]
    axes[2].plot(p.year, 100 * (np.exp(p.log_premium) - 1), color=st["color"], ls=st["ls"],
                 marker=st["marker"], ms=2.5, lw=0.8)
ru = np.exp(S["prem2030_ru"]) - 1
fc = {"RUS": (np.median(ru), *np.percentile(ru, [10, 90]))}
for iso, (a, c, b) in [("CHN", (0.24, 0.45, 0.68)), ("ZAF", (0.70, 0.99, 1.50))]:
    t = np.random.default_rng(1).triangular(a, c, b, 200000)
    fc[iso] = (np.median(t), *np.percentile(t, [10, 90]))
for k, (iso, (m, lo, hi)) in enumerate(fc.items()):
    st = STY[iso]
    axes[2].errorbar(2030 + (k - 1) * 1.1, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                     marker=st["marker"], ms=3.5, mfc="white", capsize=1.5, lw=0.9)
for ax in axes[:2]:
    ax.axvspan(2025.5, 2030.5, color="0.92", lw=0, zorder=0)
    ax.set_xlim(2000, 2030.5)
axes[2].axvspan(2028.4, 2031.6, color="0.92", lw=0, zorder=0)
axes[2].set_xlim(1996, 2031.8)
for ax in axes:
    ax.set_xticks([2000, 2015, 2030])
    ax.tick_params(labelsize=6.5)
axes[0].set_title("а) Безработица, %", fontsize=7.5)
axes[1].set_title("б) Рост ВВП на душу, %", fontsize=7.5)
axes[2].set_title("в) Премия Π, %", fontsize=7.5)
axes[1].axhline(0, color="0.5", lw=0.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.4, fontsize=7)
commas(fig)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
fig.savefig(FIG / "fig1_inputs.png")
plt.close(fig)

# ---------------------------------------------------------------- Рис. 1: прогнозы входов модели
fig = plt.figure(figsize=(W_IN, 3.1))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
axd = fig.add_subplot(gs[1, :])
for iso, st in STY.items():
    d = macro[macro.iso == iso]
    um, gm = R["ts"][iso]["unemp_model"], R["ts"][iso]["growth_model"]
    ax = axes[0]
    ax.plot(d.year, d.unemp, color=st["color"], ls=st["ls"], lw=0.9, label=st["label"])
    ax.plot(np.r_[2025, H], np.r_[d.unemp.iloc[-1], um["forecast"]], color=st["color"], ls=st["ls"], lw=1.5)
    fan(ax, H, um["lo80"], um["hi80"], st["color"])
    g = 100 * np.log(d.gdppc_ppp).diff()
    ax = axes[1]
    ax.plot(d.year.iloc[1:], g.iloc[1:], color=st["color"], ls=st["ls"], lw=0.9)
    ax.plot(np.r_[2025, H], np.r_[g.iloc[-1], gm["forecast"]], color=st["color"], ls=st["ls"], lw=1.5)
    fan(ax, H, gm["lo80"], gm["hi80"], st["color"])
    p = prem[prem.iso == iso]
    axes[2].plot(p.year, 100 * (np.exp(p.log_premium) - 1), color=st["color"], ls=st["ls"],
                 marker=st["marker"], ms=2.5, lw=0.8)
ru = np.exp(S["prem2030_ru"]) - 1
fc = {"RUS": (np.median(ru), *np.percentile(ru, [10, 90]))}
for iso, (a, c, b) in [("CHN", (0.24, 0.45, 0.68)), ("ZAF", (0.70, 0.99, 1.50))]:
    t = np.random.default_rng(1).triangular(a, c, b, 200000)
    fc[iso] = (np.median(t), *np.percentile(t, [10, 90]))
for k, (iso, (m, lo, hi)) in enumerate(fc.items()):
    st = STY[iso]
    axes[2].errorbar(2030 + (k - 1) * 1.1, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                     marker=st["marker"], ms=3.5, mfc="white", capsize=1.5, lw=0.9)
for ax in axes[:2]:
    ax.axvspan(2025.5, 2030.5, color="0.92", lw=0, zorder=0)
    ax.set_xlim(2000, 2030.5)
axes[2].axvspan(2028.4, 2031.6, color="0.92", lw=0, zorder=0)
axes[2].set_xlim(1996, 2031.8)
for ax in axes:
    ax.set_xticks([2000, 2015, 2030])
    ax.tick_params(labelsize=6.5)
axes[0].set_title("а) Безработица, %", fontsize=7.5)
axes[1].set_title("б) Рост ВВП на душу, %", fontsize=7.5)
axes[2].set_title("в) Премия Π, %", fontsize=7.5)
axes[1].axhline(0, color="0.5", lw=0.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.4, fontsize=7)
retro = pd.DataFrame(R["retro"])
scen = {"RUS": "RUS_budget", "CHN": "CHN_public", "ZAF": "ZAF_self"}
for k, (iso, st) in enumerate(STY.items()):
    r = retro[retro.iso == iso]
    axd.plot(r.year, 100 * r.irr_exante, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.5, lw=0.9)
    irr = S["mc"][scen[iso]]["irr_x"]
    irr = irr[~np.isnan(irr)]
    m, lo, hi = np.median(irr), *np.percentile(irr, [10, 90])
    axd.errorbar(2030 + (k - 1) * 0.9, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                 marker=st["marker"], ms=3.5, mfc="white", capsize=1.5, lw=0.9)
irr = S["mc"]["RUS_paid_top"]["irr_x"]
m, lo, hi = np.median(irr), *np.percentile(irr, [10, 90])
axd.errorbar(2031.4, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=STY["RUS"]["color"],
             marker="D", ms=3, mfc="white", capsize=1.5, lw=0.8)
axd.annotate("Россия, ведущий вуз", (2031.2, 100 * m), xytext=(2021.5, 4.2), fontsize=6.3,
             arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
axd.axhline(3, color="0.5", lw=0.6, ls="-.")
axd.text(2003, 1.0, "ставка дисконтирования 3%", fontsize=6.3, color="0.35")
axd.axvspan(2028.3, 2032.2, color="0.92", lw=0, zorder=0)
axd.set_xlim(1997, 2032.2)
axd.set_ylim(0, 30)
axd.set_xticks([2000, 2005, 2010, 2015, 2020, 2025, 2030])
axd.tick_params(labelsize=6.5)
axd.set_title("г) IRR абитуриента в условиях отдельных лет и прогноз для когорты 2026 г., %", fontsize=7.5)
commas(fig)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.6, h_pad=0.6)
fig.savefig(FIG / "fig1_forecasts.png")
plt.close(fig)

# ---------------------------------------------------------------- Рис. 2: IRR абитуриента
retro = pd.DataFrame(R["retro"])
fig, ax = plt.subplots(figsize=(W_IN, 1.75))
scen = {"RUS": "RUS_budget", "CHN": "CHN_public", "ZAF": "ZAF_self"}
for k, (iso, st) in enumerate(STY.items()):
    r = retro[retro.iso == iso]
    ax.plot(r.year, 100 * r.irr_exante, color=st["color"], ls=st["ls"], marker=st["marker"],
            ms=3, lw=0.9, label=st["label"])
    irr = S["mc"][scen[iso]]["irr_x"]
    irr = irr[~np.isnan(irr)]
    m, lo, hi = np.median(irr), *np.percentile(irr, [10, 90])
    x = 2030 + (k - 1) * 0.9
    ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                marker=st["marker"], ms=4, mfc="white", capsize=2, lw=1)
irr = S["mc"]["RUS_paid_top"]["irr_x"]
m, lo, hi = np.median(irr), *np.percentile(irr, [10, 90])
ax.errorbar(2031.3, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=STY["RUS"]["color"],
            marker="D", ms=3.5, mfc="white", capsize=2, lw=0.8)
ax.annotate("Россия,\nведущий вуз", (2031.1, 100 * m), xytext=(2024.6, 5.2), fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
ax.axhline(3, color="0.5", lw=0.6, ls="-.")
ax.text(2005, 1.3, "ставка дисконтирования 3%", fontsize=6.5, color="0.35")
ax.text(2030, 29, "прогноз\n(когорта 2026 г.)", fontsize=6.5, ha="center", va="top", color="0.3")
ax.axvspan(2028.3, 2032.2, color="0.92", lw=0, zorder=0)
ax.set_xlim(1997, 2032.2)
ax.set_ylim(0, 30)
ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025, 2030])
ax.set_ylabel("IRR, % годовых (реальная)")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.84, 1.0), ncol=3, handlelength=2.4)
commas(fig)
fig.tight_layout()
fig.savefig(FIG / "fig2_irr.png")
plt.close(fig)

print("ok")
