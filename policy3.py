"""Дополнительные расчёты для статьи (v3): калибровка безработицы, длительность программы (реформа
«базового высшего» 4–6 лет), критическая плата за обучение. Пишет out_v3/policy.json."""
import json, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import lifecycle3 as L

BASE = Path(__file__).resolve().parent
R = json.loads((BASE / "out_v3" / "results.json").read_text())
ST = pickle.load(open(BASE / "out_v3" / "state.pkl", "rb"))
out = {}

# 1) безработица молодых без вуза (до 25 лет) в модели, 2026 г.
c = {iso: next(v for v in R["central"].values() if v["iso"] == iso) for iso in ("RUS", "CHN", "ZAF")}
out["calib"] = {iso: dict(u2026=c[iso]["u"][0] * 100,
                          young_school=c[iso]["u"][0] * 100 * L.COUNTRY[iso]["k_s"] * L.COUNTRY[iso]["m"][0],
                          school_prime=c[iso]["u"][0] * 100 * L.COUNTRY[iso]["k_s"],
                          grad_prime=c[iso]["u"][0] * 100 * L.COUNTRY[iso]["k_g"]) for iso in c}


# 2) длительность программы S = 4, 5, 6 лет (Россия), Монте-Карло на тех же розыгрышах
def mc_S(sc, S, prem_shift=0.0):
    iso, cost = "RUS", R["central"][sc]["cost"]
    P = dict(L.COUNTRY[iso], S=S)
    D = ST["draws"][iso]
    ulr = R["central"][sc]["u_longrun"]
    irr_x, pb = [], []
    for i in range(len(D["prem"])):
        W, u = L.macro_paths(ST["G"][iso][i], ST["U"][iso][i], ulr)
        g, x, _ = L.flows(iso, cost, D["prem"][i] + prem_shift, W, u, D["compl"][i], P=P)
        irr_x.append(L.irr(x)); pb.append(L.payback(g, S))
    return float(np.nanmedian(irr_x)), float(np.nanmedian(pb))


out["duration"] = {sc: {S: mc_S(sc, S) for S in (4, 5, 6)} for sc in ("RUS_budget", "RUS_paid_avg")}


# 3) критическая годовая плата: NPV абитуриента = 0 (центральный сценарий), r = 3% и 8%
def crit_cost(sc, r, S=None):
    cc = R["central"][sc]
    P = dict(L.COUNTRY[cc["iso"]]) if S is None else dict(L.COUNTRY[cc["iso"]], S=S)
    f = lambda k: L.npv(L.flows(cc["iso"], k, cc["prem_log"], np.array(cc["W"]), np.array(cc["u"]), cc["compl"], P=P)[1], r)
    return brentq(f, 0, 1e8)


out["crit_cost"] = {sc: {str(r): crit_cost(sc, r) for r in (0.03, 0.08)}
                    for sc in ("RUS_budget", "CHN_public", "ZAF_self")}
out["crit_cost_wage_months"] = {sc: {r: v / (12 * L.COUNTRY[R["central"][sc]["iso"]]["w0"]) * 12 for r, v in d.items()}
                                for sc, d in out["crit_cost"].items()}
(BASE / "out_v3" / "policy.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


# 4) какая премия нужна, чтобы при S = 5, 6 сохранить IRR абитуриента S = 4 (центральный сценарий, бюджет)
def irr_c(sc, S, prem):
    cc = R["central"][sc]
    P = dict(L.COUNTRY["RUS"], S=S)
    return L.irr(L.flows("RUS", cc["cost"], prem, np.array(cc["W"]), np.array(cc["u"]), cc["compl"], P=P)[1])


cc = R["central"]["RUS_budget"]
target = irr_c("RUS_budget", 4, cc["prem_log"])
out["prem_needed"] = {S: float(np.exp(brentq(lambda p: irr_c("RUS_budget", S, p) - target, 0.1, 2.0)) - 1)
                      for S in (5, 6)}
out["prem_needed"]["base"] = float(np.exp(cc["prem_log"]) - 1)
(BASE / "out_v3" / "policy.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(out["prem_needed"])


# 5) ЮАР: проживание в общежитии (+52 000 рандов в год, норматив NSFAS 2025) — Монте-Карло
def mc_cost(sc, cost):
    iso = R["central"][sc]["iso"]
    D = ST["draws"][iso]
    ulr = R["central"][sc]["u_longrun"]
    irr_x, pb = [], []
    for i in range(len(D["prem"])):
        W, u = L.macro_paths(ST["G"][iso][i], ST["U"][iso][i], ulr)
        g, x, _ = L.flows(iso, cost, D["prem"][i], W, u, D["compl"][i])
        irr_x.append(L.irr(x)); pb.append(L.payback(g, L.COUNTRY[iso]["S"]))
    return float(np.nanmedian(irr_x)), float(np.nanmedian(pb))


out["za_residence"] = dict(base=mc_cost("ZAF_self", R["central"]["ZAF_self"]["cost"]),
                           residence=mc_cost("ZAF_self", R["central"]["ZAF_self"]["cost"] + 52_000))
(BASE / "out_v3" / "policy.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(out["za_residence"])
