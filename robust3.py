"""Проверки устойчивости центрального расчёта v3 (используются в тексте статьи).

1) Замена прогнозов ARIMA на 2026–2030 гг. прогнозами МВФ (WEO, октябрь 2026 г.): безработица (LUR) для
   трёх стран; рост реального ВВП на душу — только для Китая (NGDP_RPCH с поправкой на численность
   населения LP), для России и ЮАР рост остаётся по ARIMA. Долгосрочные уровни не меняются.
2) «Торнадо» чувствительности IRR абитуриента к отдельным параметрам при прочих центральных:
   премия — границы диапазона (Россия: 10-й и 90-й процентили прогноза 2030 г.; Китай, ЮАР — границы
   треугольного распределения); прямые расходы — ±50%; доля окончивших — границы диапазона;
   безработица — весь путь ×1,5 и ×0,5.
"""
import json
from pathlib import Path

import numpy as np
import lifecycle3 as L

BASE = Path(__file__).resolve().parent
OUT = BASE / "out_v3"
R = json.loads((OUT / "results.json").read_text())

IMF_U = {"RUS": [2.2, 2.4, 3.3, 3.6, 3.8], "CHN": [5.1, 5.1, 5.1, 5.1, 5.1], "ZAF": [33.2, 33.0, 32.7, 32.4, 32.0]}
CN_PC = [4.735356162692894, 4.238634446030254, 4.2620484885760535, 3.984969155813811, 3.605039182749503]


def run(sc, prem=None, cost=None, compl=None, W=None, u=None):
    c = R["central"][sc]
    iso = c["iso"]
    W = np.array(c["W"]) if W is None else W
    u = np.array(c["u"]) if u is None else u
    g, x, _ = L.flows(iso, c["cost"] if cost is None else cost, c["prem_log"] if prem is None else prem,
                      W, u, c["compl"] if compl is None else compl)
    return g, x, L.COUNTRY[iso]["S"]


rob = {}
for sc, c in R["central"].items():
    iso = c["iso"]
    g5 = np.array(CN_PC) if iso == "CHN" else np.array(c["g5"])
    W, u = L.macro_paths(g5, np.array(IMF_U[iso]), c["u_longrun"])
    g, x, S = run(sc, W=W, u=u)
    g0, x0, _ = run(sc)
    rob[sc] = dict(irr_x=L.irr(x), pb_g=L.payback(g, S), base_irr_x=L.irr(x0), base_pb=L.payback(g0, S))
(OUT / "robust_imf.json").write_text(json.dumps(dict(imf_u=IMF_U, cn_pc=CN_PC, rob=rob), indent=1))

prem_rng = {"RUS": (R["premium"]["ru_2030"]["p10"], R["premium"]["ru_2030"]["p90"])}
for iso in ("CHN", "ZAF"):
    lo, _, hi = R["params"]["prem"][iso][1]
    prem_rng[iso] = (np.log(1 + lo), np.log(1 + hi))
tor = {}
for sc, c in R["central"].items():
    if sc == "ZAF_nsfas":
        continue
    iso = c["iso"]
    lo_c, hi_c = L.COUNTRY[iso]["compl"]
    irr = lambda **kw: L.irr(run(sc, **kw)[1])
    tor[sc] = dict(base=irr(),
                   prem=[irr(prem=v) for v in prem_rng[iso]],
                   cost=[irr(cost=c["cost"] * 1.5), irr(cost=c["cost"] * 0.5)],
                   compl=[irr(compl=lo_c), irr(compl=hi_c)],
                   u=[irr(u=np.array(c["u"]) * 1.5), irr(u=np.array(c["u"]) * 0.5)])
(OUT / "tornado_v2.json").write_text(json.dumps(tor, indent=1))
for sc in tor:
    print(sc, {k: [round(100 * v, 2) for v in (vv if isinstance(vv, list) else [vv])] for k, vv in tor[sc].items()},
          "| МВФ:", round(100 * rob[sc]["irr_x"], 2), "vs", round(100 * rob[sc]["base_irr_x"], 2))
