"""Неоднородность окупаемости (v3): направления подготовки, престиж вуза, разложение выгоды диплома.

1) Направления подготовки. Премия направления f: pi_f = pi + ln(w_f / w_avg), где w_f / w_avg — отношение
   средней зарплаты выпускников направления к средней по всем выпускникам того же года (Россия — Роструд,
   выпуск 2018 г., зарплата 2021 г., доклад НИУ ВШЭ 2022; Китай — MyCOS, выпуск 2023 г. (менеджмент — 2022 г.),
   через полгода после выпуска). Допущение: относительный разрыв сохраняется на протяжении карьеры.
   Медицина в России — специалитет 6 лет. Монте-Карло на тех же 10 000 розыгрышах, что и основной расчёт.
2) Престиж вуза (Россия): минимальная прибавка к заработку выпускника ведущего вуза, при которой платное
   обучение в нём (450 тыс. руб. в год) даёт абитуриенту ту же NPV, что и бюджетное место / платное
   обучение по средней стоимости (270 тыс.) в обычном вузе; центральный сценарий, r = 3% и 8%.
3) Симметричное (шепли) разложение приведённого прироста дохода после выпуска на вклад зарплаты и
   вклад занятости:  p_H w_H − p_S w_S = (w_H − w_S)(p_H + p_S)/2 + (p_H − p_S)(w_H + w_S)/2,
   где w — доход занятого после налога, p — вероятность занятости. Центральный сценарий, r = 3%.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import lifecycle3 as L

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_v3"
R = json.loads((OUT / "results.json").read_text())
ST = pickle.load(open(OUT / "state.pkl", "rb"))

FIELDS = {
    "RUS": dict(avg=55.4, fields=[  # тыс. руб. в месяц, 2021 г., бакалавры и специалисты выпуска 2018 г.
        ("Математика и компьютерные науки", 102.1, 4),
        ("Инженерное дело и технологии", 65.8, 4),
        ("Медицина (специалитет)", 66.2, 6),
        ("Экономика и управление", 52.4, 4),
        ("Образование и педагогика", 42.2, 4)]),
    "CHN": dict(avg=None, fields=[  # юаней в месяц через полгода после выпуска; (значение, среднее той же когорты)
        ("Инженерные науки", (6709, 6050), 4),
        ("Экономика", (6088, 6050), 4),
        ("Естественные науки", (5964, 6050), 4)]),
}
SCEN = {"RUS": [("budget", 0.0), ("paid_avg", 270_000.0)], "CHN": [("public", 20_000.0)]}


def mc(iso, cost, shift, S):
    P = dict(L.COUNTRY[iso], S=S)
    D = ST["draws"][iso]
    sc0 = next(k for k, v in R["central"].items() if v["iso"] == iso)
    ulr = R["central"][sc0]["u_longrun"]
    ix, ig, pb = [], [], []
    for i in range(len(D["prem"])):
        W, u = L.macro_paths(ST["G"][iso][i], ST["U"][iso][i], ulr)
        g, x, _ = L.flows(iso, cost, D["prem"][i] + shift, W, u, D["compl"][i], P=P)
        ix.append(L.irr(x)); ig.append(L.irr(g)); pb.append(L.payback(g, S))
    q = lambda a: dict(median=float(np.nanmedian(a)), p10=float(np.nanpercentile(a, 10)),
                       p90=float(np.nanpercentile(a, 90)))
    return dict(irr_exante=q(ix), irr_grad=q(ig), payback_grad=q(pb))


out = {"fields": {}}
for iso, spec in FIELDS.items():
    rows = []
    for name, val, S in spec["fields"]:
        ratio = val / spec["avg"] if spec["avg"] else val[0] / val[1]
        res = {"name": name, "ratio": ratio, "S": S, "prem_central":
               float(np.exp(next(v for v in R["central"].values() if v["iso"] == iso)["prem_log"] + np.log(ratio)) - 1)}
        for lab, cost in SCEN[iso]:
            res[lab] = mc(iso, cost, np.log(ratio), S)
        rows.append(res)
        print(iso, name, round(ratio, 3), {lab: round(100 * res[lab]["irr_exante"]["median"], 1) for lab, _ in SCEN[iso]},
              {lab: round(res[lab]["payback_grad"]["median"], 1) for lab, _ in SCEN[iso]})
    out["fields"][iso] = rows

# ---------------------------------------------------------------- 2) престиж вуза (Россия)
c = R["central"]["RUS_budget"]
W, u = np.array(c["W"]), np.array(c["u"])
npv_x = lambda cost, prem, r: L.npv(L.flows("RUS", cost, prem, W, u, c["compl"])[1], r)
out["prestige"] = {}
for base_lab, base_cost in [("budget", 0.0), ("paid_avg", 270_000.0)]:
    for r in (0.03, 0.08):
        target = npv_x(base_cost, c["prem_log"], r)
        d = brentq(lambda s: npv_x(450_000.0, c["prem_log"] + s, r) - target, 0.0, 3.0)
        out["prestige"][f"{base_lab}_{r}"] = float(np.exp(d) - 1)
print("prestige", out["prestige"])

# ---------------------------------------------------------------- 3) разложение: зарплата vs занятость
out["decomp"] = {}
for sc in ("RUS_budget", "CHN_public", "ZAF_self"):
    c = R["central"][sc]
    iso = c["iso"]
    P = L.COUNTRY[iso]
    W, u = np.array(c["W"]), np.array(c["u"])
    off = {"unemployment": False}
    wS = L.income(P, W, u, 0.0, 0, P["k_s"], iso, off)
    wH = L.income(P, W, u, c["prem_log"], P["S"], P["k_g"], iso, off)
    m = L.age_mult(P["m"])
    pS = 1 - np.clip(u * P["k_s"] * m, 0, 0.95)
    pH = 1 - np.clip(u * P["k_g"] * m, 0, 0.95)
    t = np.arange(L.T)
    post = t >= P["S"]
    disc = (1.03) ** -t
    wage = np.sum(disc * post * (wH - wS) * (pH + pS) / 2)
    emp = np.sum(disc * post * (pH - pS) * (wH + wS) / 2)
    out["decomp"][sc] = dict(wage=float(wage), emp=float(emp), emp_share=float(emp / (wage + emp)),
                             pS_prime=float(np.mean(pS[(L.AGES >= 25) & (L.AGES < 55)])),
                             pH_prime=float(np.mean(pH[(L.AGES >= 25) & (L.AGES < 55)])))
    print(sc, "доля занятости в приросте дохода", round(100 * emp / (wage + emp), 1), "%")

(OUT / "hetero.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- 3б) безработица и упущенный заработок
for sc in ("RUS_budget", "CHN_public", "ZAF_self"):
    c = R["central"][sc]
    iso = c["iso"]
    P = L.COUNTRY[iso]
    W, u = np.array(c["W"]), np.array(c["u"])
    yS_u = L.income(P, W, u, 0.0, 0, P["k_s"], iso)
    yS_f = L.income(P, W, u, 0.0, 0, P["k_s"], iso, {"unemployment": False})
    t = np.arange(L.T)
    st = t < P["S"]
    disc = (1.03) ** -t
    out["decomp"][sc]["forgone_ratio"] = float(np.sum(disc * st * yS_u) / np.sum(disc * st * yS_f))
    out["decomp"][sc]["forgone_pv"] = float(np.sum(disc * st * yS_u))
    out["decomp"][sc]["gain_pv"] = float(out["decomp"][sc]["wage"] + out["decomp"][sc]["emp"])
    print(sc, "упущенный заработок с учётом безработицы / при полной занятости:",
          round(out["decomp"][sc]["forgone_ratio"], 2))
(OUT / "hetero.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
