"""Единая модель для трёх случаев (v3): страны различаются только значениями шести групп параметров.

Модель переписана в относительных единицах (месячная зарплата выпускника школы w0 = 1), поэтому параметры
можно переносить между странами:
  prem   — премия (логарифм), центральный сценарий;
  cost   — прямые расходы K / w0 и длительность программы S;
  labour — путь общей безработицы, относительные риски k_S, k_H и возрастные множители;
  growth — путь роста реальной заработной платы W_t;
  tax    — налоговая шкала страны (применяется к доходу в её валюте: y · w0_страны);
  compl  — доля окончивших p.
Разность IRR абитуриента двух стран раскладывается на вклады групп по Шепли (точно, 2^6 комбинаций).
Сначала проверяется, что модель в относительных единицах воспроизводит IRR основного расчёта.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

import lifecycle3 as L

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_v3"
R = json.loads((OUT / "results.json").read_text())
GROUPS = ["prem", "cost", "labour", "growth", "tax", "compl"]
SC = {"RUS": "RUS_budget", "CHN": "CHN_public", "ZAF": "ZAF_self"}


def params(sc):
    c = R["central"][sc]
    iso = c["iso"]
    P = L.COUNTRY[iso]
    return dict(prem=c["prem_log"], cost=(c["cost"] / P["w0"], P["S"]),
                labour=(np.array(c["u"]), P["k_s"], P["k_g"], P["m"]), growth=np.array(c["W"]),
                tax=iso, compl=c["compl"])


def income_rel(par, prem, start, k):
    t = np.arange(L.T)
    e = t - start
    u, _, _, m = par["labour"]
    emp = 1 - np.clip(u * k * L.age_mult(m), 0, 0.95)
    gross = 12 * par["growth"] * np.exp(prem) * np.exp(L.B1 * e + L.B2 * e ** 2)
    w0 = L.COUNTRY[par["tax"]]["w0"]
    net = gross - L.tax(par["tax"], gross * w0) / w0
    return np.where(e >= 0, emp * net, 0.0)


def flows_rel(par):
    (K, S), (_, ks, kg, _) = par["cost"], par["labour"]
    t = np.arange(L.T)
    ys = income_rel(par, 0.0, 0, ks)
    yg = income_rel(par, par["prem"], S, kg)
    d = int(np.ceil(S / 2))
    yd = income_rel(par, 0.0, d, ks)
    grad = yg - np.where(t < S, K, 0.0) - ys
    drop = yd - np.where(t < d, K, 0.0) - ys
    return grad, par["compl"] * grad + (1 - par["compl"]) * drop


def irr_x(par):
    return L.irr(flows_rel(par)[1])


out = {"check": {}, "shapley": {}, "table": {}}
for iso, sc in SC.items():
    c = R["central"][sc]
    g, x, _ = L.flows(iso, c["cost"], c["prem_log"], np.array(c["W"]), np.array(c["u"]), c["compl"])
    out["check"][sc] = (L.irr(x), irr_x(params(sc)))
    assert abs(out["check"][sc][0] - out["check"][sc][1]) < 1e-9, out["check"][sc]
print("проверка совпадения:", {k: [round(100 * v, 4) for v in vv] for k, vv in out["check"].items()})


def shapley(a, b):
    """Вклады групп в IRR(b) − IRR(a): b — страна назначения, a — база."""
    pa, pb = params(a), params(b)
    n = len(GROUPS)
    cache = {}

    def val(subset):
        key = frozenset(subset)
        if key not in cache:
            p = {g: (pb[g] if g in key else pa[g]) for g in GROUPS}
            cache[key] = irr_x(p)
        return cache[key]
    phi = {}
    for g in GROUPS:
        rest = [h for h in GROUPS if h != g]
        tot = 0.0
        for r in range(n):
            for sub in itertools.combinations(rest, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += w * (val(set(sub) | {g}) - val(set(sub)))
        phi[g] = tot
    return dict(total=val(set(GROUPS)) - val(set()), base=val(set()), target=val(set(GROUPS)), phi=phi)


for a, b in [("RUS_budget", "CHN_public"), ("RUS_budget", "ZAF_self"), ("CHN_public", "ZAF_self")]:
    s = shapley(a, b)
    out["shapley"][f"{b}-{a}"] = s
    print(f"{b} − {a}: {100 * s['total']:+.2f} п.п. =", {g: round(100 * v, 2) for g, v in s["phi"].items()},
          "сумма", round(100 * sum(s["phi"].values()), 2))

# параметры для таблицы «три случая»
for iso, sc in SC.items():
    c = R["central"][sc]
    P = L.COUNTRY[iso]
    e = 13  # стаж 13 лет (35 лет) у выпускника 4-летней программы
    gross = 12 * P["w0"] * np.exp(c["prem_log"]) * math.exp(L.B1 * e + L.B2 * e ** 2)
    out["table"][iso] = dict(prem=math.exp(c["prem_log"]) - 1, cost_rel=c["cost"] / (12 * P["w0"]), S=P["S"],
                             compl=c["compl"], u2026=c["u"][0] * 100, k_s=P["k_s"], k_g=P["k_g"],
                             g_mean=float(np.mean(c["g5"])), tax_avg=float(L.tax(iso, gross) / gross))
print(json.dumps(out["table"], ensure_ascii=False, indent=1))
(OUT / "unify.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
