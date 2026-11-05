"""
Моделирование динамики и прогнозирование окупаемости высшего образования:
Россия, Китай, ЮАР.

Блоки:
  1. Временные ряды макро-драйверов (безработица, рост реального ВВП на душу):
     тесты ADF/KPSS, тест Чоу на структурный сдвиг, выбор ARIMA по AICc,
     скользящая проверка на отложенной выборке, прогноз 2026–2030 и симуляция путей.
  2. Массовизация: логистическая (S-образная) модель охвата высшим образованием (Китай).
  3. Динамика образовательной премии: модель частичной адаптации (NLS) для России,
     регрессия премии на охват высшим образованием (Россия; Россия+Китай с FE).
  4. Модель жизненного цикла (профиль Минсера) для когорты 2026 г.:
     NPV, IRR, срок окупаемости; Монте-Карло; ретроспектива IRR; анализ чувствительности.

Версия 3 (4 ноября 2026): прогрессивные шкалы налогов, доля окончивших в ЮАР 50–72%.
Версия 3 (ноябрь 2026): прогрессивные налоги, ядро в lifecycle3.py (версия 2: ядро вынесено в lifecycle.py); добавлен сценарий
средней платы в РФ; общие случайные числа для сценариев страны; чувствительность NPV к ставке.
Запуск: python model_v3.py  (результаты в out_v3/)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DATA, OUT, FIG = BASE / "data", BASE / "out_v3", BASE / "figures_v3"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RNG = np.random.default_rng(2026)
N_SIM = 10_000
H = [2026, 2027, 2028, 2029, 2030]
COUNTRIES = ["RUS", "CHN", "ZAF"]
NAMES = {"RUS": "Россия", "CHN": "Китай", "ZAF": "ЮАР"}

macro = pd.read_csv(DATA / "macro.csv")
ger = pd.read_csv(DATA / "ger.csv")
prem = pd.read_csv(DATA / "premium.csv")


def series(iso, col, start=None):
    d = macro[macro.iso == iso].set_index("year")[col]
    if start:
        d = d.loc[start:]
    d.index = pd.PeriodIndex(d.index, freq="Y")
    return d


# ---------------------------------------------------------------------------
# 1. Временные ряды
# ---------------------------------------------------------------------------
def unit_root(x):
    x = x.dropna()
    adf_p = adfuller(x, autolag="AIC", regression="c")[1]
    try:
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    return adf_p, kpss_p


def fit_best(y, candidates, exog=None):
    """Перебор спецификаций, выбор по AICc."""
    best = None
    rows = []
    for order, trend in candidates:
        try:
            res = ARIMA(y, order=order, trend=trend, exog=exog).fit()
        except Exception:
            continue
        rows.append((order, trend, res.aicc))
        if best is None or res.aicc < best[2].aicc:
            best = (order, trend, res)
    return best, rows


def rolling_rmse(y, order, trend, origins, exog=None, naive="last"):
    """Скользящий прогноз на 1 шаг вперёд: модель vs наивный прогноз."""
    err_m, err_n = [], []
    for o in origins:
        train = y.loc[:str(o)]
        test_val = y.loc[str(o + 1)]
        ex_tr = exog.loc[:str(o)] if exog is not None else None
        ex_te = exog.loc[[pd.Period(o + 1, "Y")]] if exog is not None else None
        try:
            res = ARIMA(train, order=order, trend=trend, exog=ex_tr).fit()
            f = res.forecast(1, exog=ex_te).iloc[0]
        except Exception:
            f = train.iloc[-1]
        err_m.append(f - test_val)
        err_n.append((train.iloc[-1] if naive == "last" else train.mean()) - test_val)
    return float(np.sqrt(np.mean(np.square(err_m)))), float(np.sqrt(np.mean(np.square(err_n))))


def simulate_paths(res, steps, n, exog_f=None):
    # statsmodels ≥ 0.15 не использует глобальное зерно numpy: передаём собственный генератор
    sub = np.random.default_rng(int(RNG.integers(1e9)))
    sims = res.simulate(nsimulations=steps, repetitions=n, anchor="end", exog=exog_f, rng=sub)
    return np.asarray(sims).reshape(steps, n).T  # (n, steps)


ts_report = {}
U_PATHS, G_PATHS = {}, {}
FCST = {}

for iso in COUNTRIES:
    rep = {}
    # --- 1a. Безработица: моделируем ln(u), чтобы прогноз оставался положительным
    u = series(iso, "unemp", 2000)
    lu = np.log(u)
    rep["unemp_unit_root_level"] = unit_root(lu)
    rep["unemp_unit_root_diff"] = unit_root(lu.diff())
    cand_u = [((p, 1, q), "n") for p in range(3) for q in range(3)] + \
             [((p, 0, q), "c") for p in range(1, 3) for q in range(3)]
    (order, trend, res_u), _ = fit_best(lu, cand_u)
    rm, rn = rolling_rmse(lu, order, trend, list(range(2015, 2025)))
    lb = sm.stats.acorr_ljungbox(res_u.resid.iloc[1:], lags=[4], return_df=True)["lb_pvalue"].iloc[0]
    fc = res_u.get_forecast(len(H))
    ci80 = fc.conf_int(alpha=0.2)
    ci95 = fc.conf_int(alpha=0.05)
    rep["unemp_model"] = dict(order=order, trend=trend, aicc=res_u.aicc, ljungbox_p4=lb,
                              rmse_model=rm, rmse_naive=rn,
                              forecast=np.exp(fc.predicted_mean).round(2).tolist(),
                              lo80=np.exp(ci80.iloc[:, 0]).round(2).tolist(),
                              hi80=np.exp(ci80.iloc[:, 1]).round(2).tolist(),
                              lo95=np.exp(ci95.iloc[:, 0]).round(2).tolist(),
                              hi95=np.exp(ci95.iloc[:, 1]).round(2).tolist())
    U_PATHS[iso] = np.exp(simulate_paths(res_u, len(H), N_SIM))
    FCST[(iso, "u")] = (u, fc, ci80, ci95)

    # --- 1b. Темп роста реального ВВП на душу (ППП, пост. цены), % г/г
    gdp = series(iso, "gdppc_ppp", 2000)
    g = (100 * np.log(gdp).diff()).dropna()
    # тест Чоу на сдвиг среднего в 2009 г. (регрессия g на константу)
    pre, post = g.loc[:"2008"], g.loc["2010":]
    full = g.drop(pd.Period(2009, "Y"))
    rss_f = ((full - full.mean()) ** 2).sum()
    rss_1 = ((pre - pre.mean()) ** 2).sum()
    rss_2 = ((post - post.mean()) ** 2).sum()
    k = 1
    F = ((rss_f - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (len(full) - 2 * k))
    chow_p = 1 - stats.f.cdf(F, k, len(full) - 2 * k)
    rep["growth_chow"] = dict(mean_2001_2008=pre.mean(), mean_2010_2025=post.mean(), F=F, p=chow_p)
    # после сдвига: g на 2010–2025 с фиктивной переменной 2020 г. (пандемия)
    gpost = g.loc["2010":]
    covid = pd.Series((gpost.index.year == 2020).astype(float), index=gpost.index, name="covid")
    rep["growth_unit_root"] = unit_root(gpost)
    cand_g = [((0, 0, 0), "c"), ((1, 0, 0), "c"),
              ((0, 0, 0), "ct"), ((1, 0, 0), "ct")]
    (gorder, gtrend, res_g), rows = fit_best(gpost, cand_g, exog=covid)
    exf = pd.DataFrame({"covid": np.zeros(len(H))},
                       index=pd.period_range("2026", periods=len(H), freq="Y"))
    rmg, rng_ = rolling_rmse(gpost, gorder, gtrend, list(range(2017, 2025)), exog=covid, naive="mean")
    fcg = res_g.get_forecast(len(H), exog=exf)
    cg80 = fcg.conf_int(alpha=0.2)
    rep["growth_model"] = dict(order=gorder, trend=gtrend, aicc=res_g.aicc,
                               params={k_: float(v) for k_, v in res_g.params.items()},
                               rmse_model=rmg, rmse_mean=rng_,
                               forecast=fcg.predicted_mean.round(2).tolist(),
                               lo80=cg80.iloc[:, 0].round(2).tolist(),
                               hi80=cg80.iloc[:, 1].round(2).tolist())
    G_PATHS[iso] = simulate_paths(res_g, len(H), N_SIM, exog_f=exf)
    FCST[(iso, "g")] = (g, fcg, cg80)
    rep["u_longrun"] = float(series(iso, "unemp", 2010).mean())
    ts_report[iso] = rep


# ---------------------------------------------------------------------------
# 1в. Безработица и рост: модель по закону Оукена (ARIMAX) против случайного блуждания
#   Δln u_t = c + φ·Δln u_{t-1} + β·g_t + δ1·[2020] + δ2·[2021] + ε_t,
#   φ — инерция безработицы, β — коэффициент Оукена (на сколько % меняется безработица при +1 п.п. роста).
#   Модель выбирается для страны, если на скользящей проверке (точки отсчёта 2014–2022 гг., горизонт 3 года,
#   рост на прогнозном горизонте — среднее за 2010 г. … t) её RMSE ниже, чем у случайного блуждания.
#   В Монте-Карло безработица строится по разыгранному пути роста той же симуляции; неопределённость
#   коэффициентов учтена розыгрышем из N(b̂, V̂_HAC), шоки — N(0, σ̂²).
# ---------------------------------------------------------------------------
def okun_design(df):
    X = pd.DataFrame({"const": 1.0, "lag": df.dlu.shift(1), "g": df.g,
                      "d2020": (df.index == 2020) * 1.0, "d2021": (df.index == 2021) * 1.0}, index=df.index)
    return X.dropna()


def okun_fit(df):
    X = okun_design(df)
    return sm.OLS(df.dlu.loc[X.index], X).fit()


def okun_path(P, last_dlu, gpath, years, eps=None):
    out, prev = [], last_dlu
    for k, (yy, gg) in enumerate(zip(years, gpath)):
        x = P["const"] + P["lag"] * prev + P["g"] * gg + P["d2020"] * (yy == 2020) + P["d2021"] * (yy == 2021)
        if eps is not None:
            x += eps[k]
        out.append(x)
        prev = x
    return np.array(out)


def okun_cv(df, h=3, origins=range(2014, 2023)):
    eR, eX = [], []
    for t0 in origins:
        tr = df.loc[:t0]
        P = okun_fit(tr).params
        gf = tr.g.loc[2010:].mean()
        yrs = list(range(t0 + 1, t0 + h + 1))
        true = df.dlu.loc[t0 + 1:t0 + h].sum()
        eR.append(-true)
        eX.append(okun_path(P, tr.dlu.iloc[-1], [gf] * h, yrs).sum() - true)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    return r(eR), r(eX)


for iso in COUNTRIES:
    d = macro[macro.iso == iso].set_index("year")
    df = pd.DataFrame({"dlu": np.log(d.unemp).diff(), "g": 100 * np.log(d.gdppc_ppp).diff()}).dropna()
    fit_ = okun_fit(df)
    hac = fit_.get_robustcov_results(cov_type="HAC", maxlags=1)
    names = list(okun_design(df).columns)
    rw3, ok3 = okun_cv(df)
    rw1, ok1 = okun_cv(df, h=1)
    use = ok3 < rw3
    ts_report[iso]["unemp_compare"] = dict(
        rmse3_rw=rw3, rmse3_okun=ok3, rmse1_rw=rw1, rmse1_okun=ok1, chosen="okun" if use else "rw",
        coef={k: float(v) for k, v in zip(names, hac.params)},
        t_hac={k: float(v) for k, v in zip(names, hac.tvalues)},
        sigma=float(np.sqrt(fit_.scale)), r2=float(fit_.rsquared), n=int(fit_.nobs))
    if not use:
        continue
    P = dict(zip(names, fit_.params))
    V = np.asarray(hac.cov_params())
    u_last, dlu_last = float(d.unemp.iloc[-1]), float(df.dlu.iloc[-1])
    gpt = np.array(ts_report[iso]["growth_model"]["forecast"])
    point = u_last * np.exp(np.cumsum(okun_path(P, dlu_last, gpt, H)))
    B = RNG.multivariate_normal(fit_.params.values, V, size=N_SIM)
    E = RNG.normal(0.0, np.sqrt(fit_.scale), size=(N_SIM, len(H)))
    paths = np.empty((N_SIM, len(H)))
    for i in range(N_SIM):
        Pi = dict(zip(names, B[i]))
        paths[i] = u_last * np.exp(np.cumsum(okun_path(Pi, dlu_last, G_PATHS[iso][i], H, E[i])))
    U_PATHS[iso] = paths
    lo80, hi80 = np.percentile(paths, [10, 90], axis=0)
    lo95, hi95 = np.percentile(paths, [2.5, 97.5], axis=0)
    um = ts_report[iso]["unemp_model"]
    ts_report[iso]["unemp_model_rw"] = dict(um)
    ts_report[iso]["unemp_model"] = dict(um, order="okun", forecast=point.round(2).tolist(),
                                         lo80=lo80.round(2).tolist(), hi80=hi80.round(2).tolist(),
                                         lo95=lo95.round(2).tolist(), hi95=hi95.round(2).tolist())
for iso in COUNTRIES:
    c = ts_report[iso]["unemp_compare"]
    print(iso, "Оукен:", {k: (round(c["coef"][k], 4), round(c["t_hac"][k], 1)) for k in ("lag", "g")},
          f"RMSE(3 года) RW {c['rmse3_rw']:.3f} / Оукен {c['rmse3_okun']:.3f} -> {c['chosen']};",
          "прогноз u:", ts_report[iso]["unemp_model"]["forecast"])

# ---------------------------------------------------------------------------
# 2. Логистическая модель охвата высшим образованием (Китай)
# ---------------------------------------------------------------------------
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


gc = ger[ger.iso == "CHN"]
LB_, UB_ = [40, 0.01, 1990], [100, 1.0, 2050]   # насыщение: охват не выше 100%
popt, pcov = optimize.curve_fit(logistic, gc.year.values, gc.ger.values,
                                p0=[90, 0.15, 2015], bounds=(LB_, UB_), maxfev=20000)
resid = gc.ger.values - logistic(gc.year.values, *popt)
r2 = 1 - resid.var() / gc.ger.values.var()
boot = []
for _ in range(2000):
    yb = logistic(gc.year.values, *popt) + RNG.choice(resid, len(resid), replace=True)
    try:
        pb, _ = optimize.curve_fit(logistic, gc.year.values, yb, p0=popt,
                                   bounds=(LB_, UB_), maxfev=20000)
        boot.append(logistic(2030, *pb))
    except Exception:
        pass
ger_report = dict(K=popt[0], r=popt[1], t0=popt[2], R2=r2,
                  se=np.sqrt(np.diag(pcov)).tolist(),
                  ger2030=logistic(2030, *popt),
                  ger2030_ci90=np.percentile(boot, [5, 95]).tolist())

# ---------------------------------------------------------------------------
# 3. Динамика образовательной премии
# ---------------------------------------------------------------------------
pr = prem[prem.iso == "RUS"].copy()
t = pr.year.values - 1998
y = pr.log_premium.values


def adj(tt, p_inf, p0, lam):
    """Частичная адаптация: p_t = p_inf + (p0 - p_inf) * exp(-lam * t)."""
    return p_inf + (p0 - p_inf) * np.exp(-lam * tt)


pa, pc = optimize.curve_fit(adj, t, y, p0=[0.3, 0.52, 0.15], maxfev=20000)
res_nls = y - adj(t, *pa)
n = len(y)
aic_nls = n * np.log((res_nls ** 2).sum() / n) + 2 * 3
lin = sm.OLS(y, sm.add_constant(t)).fit()
aic_lin = n * np.log((lin.resid ** 2).sum() / n) + 2 * 2
r2_nls = 1 - (res_nls ** 2).sum() / ((y - y.mean()) ** 2).sum()
# регрессия премии на охват ВО (Россия; 2010 г. — интерполяция охвата) и панель Россия+Китай с FE
gr = ger.set_index(["iso", "year"]).ger.to_dict()
gr[("RUS", 2010)] = (gr[("RUS", 2009)] + gr[("RUS", 2011)]) / 2
pr["ger"] = [gr.get(("RUS", yy), np.nan) for yy in pr.year]
m_ger = sm.OLS(pr.log_premium, sm.add_constant(pr.ger)).fit(cov_type="HC1")
aic_ger = n * np.log((m_ger.resid ** 2).sum() / n) + 2 * 2
pc_ = prem[prem.iso.isin(["RUS", "CHN"])].copy()
pc_["ger"] = [gr.get((i, yy), np.nan) for i, yy in zip(pc_.iso, pc_.year)]
X = pd.get_dummies(pc_.iso, drop_first=True, dtype=float)
X["ger"] = pc_.ger
m_fe = sm.OLS(pc_.log_premium, sm.add_constant(X)).fit(cov_type="HC1")

# прогноз премии РФ на 2030 г.: охват стабилизировался (2014–2019 гг. ~81–85%),
# сценарий охвата 2030 г. ~ U(75; 85); неопределённость параметров + остатка
GER_RU_2030 = (75, 85)
beta = RNG.multivariate_normal(m_ger.params.values, m_ger.cov_params().values, N_SIM)
s_res = np.sqrt(m_ger.ssr / m_ger.df_resid)
g30 = RNG.uniform(*GER_RU_2030, N_SIM)
prem2030_ru = beta[:, 0] + beta[:, 1] * g30 + RNG.normal(0, s_res, N_SIM)
# справочно: следствие эластичности FE для Китая (охват 2018 -> 2030)
prem_report = dict(
    nls=dict(p_inf=pa[0], p0=pa[1], lam=pa[2], se=np.sqrt(np.diag(pc)).tolist(),
             R2=r2_nls, AIC=aic_nls, half_life=np.log(2) / pa[2]),
    linear=dict(b0=lin.params[0], b1=lin.params[1], R2=lin.rsquared, AIC=aic_lin,
                fc2030=lin.params[0] + lin.params[1] * 32),
    ger_model=dict(a=m_ger.params["const"], AIC=aic_ger, s_res=s_res, GER_2030=GER_RU_2030),
    ru_2030=dict(median=float(np.median(prem2030_ru)),
                 p10=float(np.percentile(prem2030_ru, 10)),
                 p90=float(np.percentile(prem2030_ru, 90)),
                 pct_median=float(np.exp(np.median(prem2030_ru)) - 1)),
    ger_reg_ru=dict(b=m_ger.params["ger"], se=m_ger.bse["ger"], p=m_ger.pvalues["ger"],
                    a=m_ger.params["const"], R2=m_ger.rsquared, n=int(m_ger.nobs)),
    cn_fe_implied=dict(ger2018=gr[("CHN", 2018)], ger2030=float(logistic(2030, *popt)),
                       dlog=float(m_fe.params["ger"] * (logistic(2030, *popt) - gr[("CHN", 2018)])),
                       prem2030_pct=float(np.exp(np.log(1.49) + m_fe.params["ger"] * (logistic(2030, *popt) - gr[("CHN", 2018)])) - 1)),
    ger_reg_fe=dict(b=m_fe.params["ger"], se=m_fe.bse["ger"], p=m_fe.pvalues["ger"],
                    R2=m_fe.rsquared, n=int(m_fe.nobs)),
)


# ---------------------------------------------------------------------------
# 4. Модель жизненного цикла (версия 3) — ядро в lifecycle3.py
# ---------------------------------------------------------------------------
import lifecycle3 as L

R_DISC = 0.03
SCEN = {  # (страна, прямые расходы в год, цены 2025 г.)
    "RUS_budget": ("RUS", 0, "Россия, бюджетное место"),
    "RUS_paid_avg": ("RUS", 270_000, "Россия, платно, средняя стоимость"),
    "RUS_paid_top": ("RUS", 450_000, "Россия, платно, ведущий вуз"),
    "CHN_public": ("CHN", 20_000, "Китай, государственный вуз"),
    "ZAF_self": ("ZAF", 74_000, "ЮАР, за счёт семьи"),   # плата 59–77 тыс. (BCom, Претория, 2025) + книги 5,7 тыс.
    "ZAF_nsfas": ("ZAF", 0, "ЮАР, поддержка NSFAS"),
}
PREM = {"RUS": ("reg", None), "CHN": ("tri", (0.24, 0.45, 0.68)), "ZAF": ("tri", (0.70, 0.99, 1.50))}
PREM_CENTRAL = {"RUS": float(np.median(prem2030_ru)), "CHN": float(np.log1p(0.45)),
                "ZAF": float(np.log1p(0.99))}
ULR = {iso: ts_report[iso]["u_longrun"] for iso in COUNTRIES}

# общие случайные числа: одни и те же розыгрыши для всех сценариев страны
DRAWS = {}
for iso in COUNTRIES:
    kind, par = PREM[iso]
    pl = RNG.choice(prem2030_ru, N_SIM) if kind == "reg" else np.log1p(RNG.triangular(*par, N_SIM))
    cp = RNG.uniform(*L.COUNTRY[iso]["compl"], N_SIM)
    DRAWS[iso] = dict(prem=pl, compl=cp)

mc = {}
for sc, (iso, cost, label) in SCEN.items():
    P = L.COUNTRY[iso]
    D = DRAWS[iso]
    rec = {k: np.empty(N_SIM) for k in ["irr_g", "irr_x", "npv_g1", "npv_g3", "npv_g5", "npv_g8", "npv_x",
                                        "pb_g", "pbd_g", "pb_x", "sc_g", "sc_x"]}
    for i in range(N_SIM):
        W, u = L.macro_paths(G_PATHS[iso][i], U_PATHS[iso][i], ULR[iso])
        g, x, _ = L.flows(iso, cost, D["prem"][i], W, u, D["compl"][i])
        rec["irr_g"][i], rec["irr_x"][i] = L.irr(g), L.irr(x)
        rec["npv_g1"][i], rec["npv_g3"][i], rec["npv_g5"][i], rec["npv_g8"][i] = L.npv(g, .01), L.npv(g, .03), L.npv(g, .05), L.npv(g, .08)
        rec["npv_x"][i] = L.npv(x, R_DISC)
        rec["pb_g"][i], rec["pbd_g"][i], rec["pb_x"][i] = L.payback(g, P["S"]), L.payback(g, P["S"], R_DISC), L.payback(x, P["S"])
        rec["sc_g"][i], rec["sc_x"][i] = L.sign_changes(g), L.sign_changes(x)
    mc[sc] = rec


def summ(a):
    nan = int(np.isnan(a).sum())
    a = a[~np.isnan(a)]
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), mean=float(np.mean(a)), n_nan=nan)


mc_report = {}
for sc, r in mc.items():
    iso = SCEN[sc][0]
    a0 = 12 * L.COUNTRY[iso]["w0"]
    mc_report[sc] = dict(label=SCEN[sc][2], cost=SCEN[sc][1],
                         irr_grad=summ(r["irr_g"]), irr_exante=summ(r["irr_x"]),
                         npv_grad_rel_r1=summ(r["npv_g1"] / a0), npv_grad_rel=summ(r["npv_g3"] / a0),
                         npv_grad_rel_r5=summ(r["npv_g5"] / a0), npv_grad_rel_r8=summ(r["npv_g8"] / a0),
                         p_npv8_pos_grad=float(np.mean(r["npv_g8"] > 0)), npv_exante_rel=summ(r["npv_x"] / a0),
                         npv_grad_abs=summ(r["npv_g3"]),
                         payback_grad=summ(r["pb_g"]), payback_disc_grad=summ(r["pbd_g"]),
                         payback_exante=summ(r["pb_x"]),
                         p_npv_pos_grad=float(np.mean(r["npv_g3"] > 0)),
                         p_npv_pos_exante=float(np.mean(r["npv_x"] > 0)),
                         max_sign_changes=int(max(r["sc_g"].max(), r["sc_x"].max())))

# центральный (детерминированный) расчёт — для сверки с Excel
central = {}
for sc, (iso, cost, label) in SCEN.items():
    P = L.COUNTRY[iso]
    g5 = np.array(ts_report[iso]["growth_model"]["forecast"])
    u5 = np.array(ts_report[iso]["unemp_model"]["forecast"])
    W, u = L.macro_paths(g5, u5, ULR[iso])
    cm = float(np.mean(P["compl"]))
    g, x, ys = L.flows(iso, cost, PREM_CENTRAL[iso], W, u, cm)
    central[sc] = dict(iso=iso, cost=cost, prem_log=PREM_CENTRAL[iso], compl=cm, g5=g5.tolist(),
                       u5=u5.tolist(), u_longrun=ULR[iso], W=W.tolist(), u=u.tolist(),
                       cf_grad=g.tolist(), cf_exante=x.tolist(), y_school=ys.tolist(),
                       irr_g=L.irr(g), irr_x=L.irr(x), npv_g=L.npv(g, R_DISC), npv_x=L.npv(x, R_DISC),
                       pb_g=L.payback(g, P["S"]), pbd_g=L.payback(g, P["S"], R_DISC))

# ретроспектива: IRR «в условиях года t»
u_hist = {"RUS": {1998: 13.261, 1999: 13.036}}
retro_cost = {"RUS": 0, "CHN": 20_000, "ZAF": 74_000}
retro = []
for _, row in prem.iterrows():
    iso, yr = row.iso, int(row.year)
    uu = u_hist.get(iso, {}).get(yr)
    if uu is None:
        uu = float(macro[(macro.iso == iso) & (macro.year == yr)].unemp.iloc[0])
    W = np.cumprod(np.full(L.T, 1.015))
    u = np.full(L.T, uu / 100)
    g, x, _ = L.flows(iso, retro_cost[iso], row.log_premium, W, u, float(np.mean(L.COUNTRY[iso]["compl"])))
    retro.append(dict(iso=iso, year=yr, irr_grad=L.irr(g), irr_exante=L.irr(x),
                      prem_pct=float(np.exp(row.log_premium) - 1), u=uu))
retro = pd.DataFrame(retro)


def conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


results = dict(ts=ts_report, ger=ger_report, premium=prem_report, mc=mc_report, central=central,
               retro=retro.to_dict("records"),
               params=dict(R=R_DISC, N_SIM=N_SIM, prem=PREM, prem_central=PREM_CENTRAL,
                           country=L.COUNTRY, tax=L.TAX, scen=SCEN, B1=L.B1, B2=L.B2))
(OUT / "results.json").write_text(json.dumps(results, default=conv, ensure_ascii=False, indent=1))
pd.to_pickle(dict(mc=mc, FCST=FCST, U=U_PATHS, G=G_PATHS, retro=retro, prem2030_ru=prem2030_ru,
                  ger_popt=popt, draws=DRAWS), OUT / "state.pkl")
for sc, r in mc_report.items():
    print(sc, {k: (round(v["median"], 4), round(v["p10"], 4), round(v["p90"], 4), v["n_nan"]) if isinstance(v, dict) else v
               for k, v in r.items() if k not in ("label",)})
print(retro.round(3).to_string())
