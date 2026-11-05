"""Графики для презентации (палитра и шрифт слайдов). Запускать после model_v3.py, hetero3.py, unify3.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
OUT, FIG, DATA = BASE / "out_v3", BASE / "slides_fig", BASE / "data"
FIG.mkdir(exist_ok=True)
R = json.loads((OUT / "results.json").read_text())
S = pd.read_pickle(OUT / "state.pkl")
UN = json.loads((OUT / "unify.json").read_text())
HT = json.loads((OUT / "hetero.json").read_text())
macro = pd.read_csv(DATA / "macro.csv")
prem = pd.read_csv(DATA / "premium.csv")

PAPER, INK, MUTED, GRID = "#F5F2EA", "#16233A", "#4A5568", "#DDD6C8"
STY = {"RUS": dict(color="#2B4C7E", ls="-", marker="o", label="Россия"),
       "CHN": dict(color="#B03A2E", ls="--", marker="s", label="Китай"),
       "ZAF": dict(color="#5D7A2E", ls=":", marker="^", label="ЮАР")}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "savefig.dpi": 200,
    "lines.linewidth": 2.4,
})
C = lambda v, d=1: (f"{(0.0 if abs(v) < 0.5 * 10 ** -d else v):.{d}f}").replace(".", ",").replace("-", "−")
COMMA = FuncFormatter(lambda v, p: (f"{v:g}").replace(".", ",").replace("-", "−"))
H = np.arange(2026, 2031)
q = lambda a: (np.nanmedian(a), *np.nanpercentile(a, [10, 90]))


def fan(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


# ------------------------------------------------ 1) прогнозы входов модели: 3 панели
fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.6))
for iso, st in STY.items():
    d = macro[macro.iso == iso]
    um, gm = R["ts"][iso]["unemp_model"], R["ts"][iso]["growth_model"]
    axes[0].plot(d.year, d.unemp, color=st["color"], ls=st["ls"], lw=1.8, label=st["label"])
    axes[0].plot(np.r_[2025, H], np.r_[d.unemp.iloc[-1], um["forecast"]], color=st["color"], ls=st["ls"], lw=3)
    fan(axes[0], H, um["lo80"], um["hi80"], st["color"])
    g = 100 * np.log(d.gdppc_ppp).diff()
    axes[1].plot(d.year.iloc[1:], g.iloc[1:], color=st["color"], ls=st["ls"], lw=1.8)
    axes[1].plot(np.r_[2025, H], np.r_[g.iloc[-1], gm["forecast"]], color=st["color"], ls=st["ls"], lw=3)
    fan(axes[1], H, gm["lo80"], gm["hi80"], st["color"])
    p = prem[prem.iso == iso]
    axes[2].plot(p.year, 100 * (np.exp(p.log_premium) - 1), color=st["color"], ls=st["ls"],
                 marker=st["marker"], ms=6, lw=1.8)
ru = np.exp(S["prem2030_ru"]) - 1
fc = {"RUS": q(ru)}
for iso, (a, c, b) in [("CHN", (0.24, 0.45, 0.68)), ("ZAF", (0.70, 0.99, 1.50))]:
    fc[iso] = q(np.random.default_rng(1).triangular(a, c, b, 200000))
for k, (iso, (m, lo, hi)) in enumerate(fc.items()):
    st = STY[iso]
    axes[2].errorbar(2030 + (k - 1) * 1.1, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                     marker=st["marker"], ms=9, mfc=PAPER, mew=2, capsize=4, lw=2)
for ax in axes[:2]:
    ax.axvspan(2025.5, 2030.5, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.set_xlim(2000, 2030.8)
axes[2].axvspan(2028.4, 2031.6, color=GRID, alpha=0.55, lw=0, zorder=0)
axes[2].set_xlim(1996, 2031.8)
for ax, t in zip(axes, ["Безработица, %", "Рост ВВП на душу, % в год", "Премия к зарплате Π, %"]):
    ax.set_xticks([2000, 2015, 2030])
    ax.set_title(t, loc="left", fontweight="semibold", pad=12)
    ax.yaxis.set_major_formatter(COMMA)
axes[1].axhline(0, color=MUTED, lw=0.8)
axes[0].text(2028, 36.5, "прогноз", ha="center", fontsize=13, color=MUTED)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=3)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
fig.savefig(FIG / "s_inputs.png")
plt.close(fig)

# ------------------------------------------------ 2) IRR абитуриента когорты 2026 г.
scen = [("RUS_budget", "Россия · бюджет", "RUS"), ("RUS_paid_avg", "Россия · средняя плата", "RUS"),
        ("RUS_paid_top", "Россия · ведущий вуз", "RUS"), ("CHN_public", "Китай · гос. вуз", "CHN"),
        ("ZAF_self", "ЮАР · за счёт семьи", "ZAF")]
fig, ax = plt.subplots(figsize=(10.4, 6.2))
for i, (sc, lab, iso) in enumerate(scen[::-1]):
    v = R["mc"][sc]["irr_exante"]
    m, lo, hi = 100 * v["median"], 100 * v["p10"], 100 * v["p90"]
    col = STY[iso]["color"]
    ax.plot([lo, hi], [i, i], color=col, lw=10, alpha=0.28, solid_capstyle="round")
    ax.plot(m, i, marker=STY[iso]["marker"], ms=17, color=col, mec=PAPER, mew=2)
    ax.text(hi + 0.7, i, C(m) + "%", va="center", fontsize=17, fontweight="semibold", color=INK)
ax.axvline(3, color=MUTED, lw=1.2, ls="-.")
ax.text(3.25, 4.45, "ставка\nдисконтирования 3%", fontsize=13, color=MUTED, va="top")
ax.set_yticks(range(len(scen)))
ax.set_yticklabels([s[1] for s in scen[::-1]], fontsize=16, color=INK)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 21)
ax.set_xlabel("IRR абитуриента, % годовых (реальная): медиана и 80%-ный интервал")
ax.xaxis.set_major_formatter(COMMA)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "s_irr.png")
plt.close(fig)

# ------------------------------------------------ 3) разложение Шепли
NAMES = {"prem": "Премия", "cost": "Затраты", "labour": "Рынок труда", "growth": "Рост зарплат",
         "tax": "Налоги", "compl": "Отчисление"}
fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.0), sharey=True)
for ax, key, title in [(axes[0], "ZAF_self-RUS_budget", "ЮАР минус Россия (бюджет)"),
                       (axes[1], "CHN_public-RUS_budget", "Китай минус Россия (бюджет)")]:
    sh = UN["shapley"][key]
    gs = list(NAMES)
    vals = [100 * sh["phi"][g] for g in gs]
    y = np.arange(len(gs))[::-1]
    cols = ["#2B6E8F" if v >= 0 else "#C2702E" for v in vals]
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + (0.35 if v >= 0 else -0.35), yy, ("+" if v >= 0.05 else "") + C(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=15, fontweight="semibold")
    ax.axvline(0, color=INK, lw=1.2)
    tot = 100 * sh["total"]
    ax.set_title(f"{title}: итого {'+' if tot >= 0 else ''}{C(tot)} п.п.", loc="left", fontweight="semibold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[g] for g in gs], fontsize=16, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(COMMA)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_xlim(-16.5, 15)
axes[1].set_xlim(-4.5, 3)
axes[0].set_xlabel("вклад в разницу IRR, п.п.")
axes[1].set_xlabel("вклад в разницу IRR, п.п.")
fig.tight_layout(w_pad=3)
fig.savefig(FIG / "s_shapley.png")
plt.close(fig)

# ------------------------------------------------ 4) направления подготовки
RU = [("Математика и компьютерные науки", "Математика и CS"), ("Инженерное дело и технологии", "Инженерия"),
      ("Медицина (специалитет)", "Медицина (6 лет)"), ("Экономика и управление", "Экономика и управление"),
      ("Образование и педагогика", "Педагогика")]
CN = [("Инженерные науки", "Инженерные науки"), ("Экономика", "Экономика"), ("Естественные науки", "Естеств. науки")]
FR = {iso: {f["name"]: f for f in rows} for iso, rows in HT["fields"].items()}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.6, 6.0), gridspec_kw=dict(width_ratios=[1.55, 1]))
y = np.arange(len(RU))[::-1]
b = [100 * FR["RUS"][k]["budget"]["irr_exante"]["median"] for k, _ in RU]
p = [100 * FR["RUS"][k]["paid_avg"]["irr_exante"]["median"] for k, _ in RU]
a1.barh(y + 0.19, b, height=0.36, color="#2B4C7E", label="бюджетное место")
a1.barh(y - 0.19, p, height=0.36, color="#9DB0CF", label="платно, 270 тыс. руб. в год")
for yy, v in zip(y, b):
    a1.text(v + 0.5, yy + 0.19, C(v) + "%", va="center", fontsize=14, fontweight="semibold")
for yy, v in zip(y, p):
    a1.text(v + 0.5, yy - 0.19, C(v) + "%", va="center", fontsize=13, color=MUTED)
a1.set_yticks(y)
a1.set_yticklabels([lab for _, lab in RU], fontsize=16, color=INK)
a1.set_xlim(0, 33)
a1.set_title("Россия", loc="left", fontweight="semibold", pad=12)
a1.legend(frameon=False, loc="lower right", fontsize=14)
yc = np.arange(len(CN))[::-1]
c = [100 * FR["CHN"][k]["public"]["irr_exante"]["median"] for k, _ in CN]
a2.barh(yc, c, height=0.5, color="#B03A2E")
for yy, v in zip(yc, c):
    a2.text(v + 0.4, yy, C(v) + "%", va="center", fontsize=14, fontweight="semibold")
a2.set_yticks(yc)
a2.set_yticklabels([lab for _, lab in CN], fontsize=16, color=INK)
a2.set_xlim(0, 33)
a2.set_ylim(-1.2, 3.2)
a2.set_title("Китай · гос. вуз", loc="left", fontweight="semibold", pad=12)
for ax in (a1, a2):
    ax.axvline(3, color=MUTED, lw=1.2, ls="-.")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(COMMA)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("IRR абитуриента, %")
fig.tight_layout(w_pad=3)
fig.savefig(FIG / "s_fields.png")
plt.close(fig)

# ------------------------------------------------ 5) траектории IRR: история и прогноз
retro = pd.DataFrame(R["retro"])
sc3 = {"RUS": "RUS_budget", "CHN": "CHN_public", "ZAF": "ZAF_self"}
fig, ax = plt.subplots(figsize=(10.6, 6.4))
for k, (iso, st) in enumerate(STY.items()):
    r = retro[retro.iso == iso]
    ax.plot(r.year, 100 * r.irr_exante, color=st["color"], ls=st["ls"], marker=st["marker"], ms=7, lw=2.2,
            label=st["label"])
    m, lo, hi = q(S["mc"][sc3[iso]]["irr_x"])
    ax.errorbar(2030 + (k - 1) * 1.0, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=st["color"],
                marker=st["marker"], ms=10, mfc=PAPER, mew=2, capsize=4, lw=2)
ax.axhline(3, color=MUTED, lw=1.2, ls="-.")
ax.text(1998.5, 1.0, "ставка дисконтирования 3%", fontsize=13, color=MUTED)
ax.axvspan(2028.2, 2031.8, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.text(2030, 28.6, "когорта\n2026 г.", ha="center", va="top", fontsize=13, color=MUTED)
ax.set_xlim(1997, 2032)
ax.set_ylim(0, 30)
ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025, 2030])
ax.set_ylabel("IRR абитуриента, %")
ax.yaxis.set_major_formatter(COMMA)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.45, 1.08), handlelength=3)
fig.tight_layout()
fig.savefig(FIG / "s_history.png")
plt.close(fig)
print("ok", sorted(p.name for p in FIG.iterdir()))
