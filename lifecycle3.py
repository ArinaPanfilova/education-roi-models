"""Ядро модели жизненного цикла (версия 3: прогрессивные шкалы налога на доходы): денежные потоки, NPV, IRR, сроки окупаемости.

Соглашения (они же описаны в методологии):
  * t = 0 — первый год обучения (2026 г., возраст 18 лет); потоки года t относятся к началу года
    и дисконтируются множителем (1 + r)^(-t), т.е. поток t = 0 не дисконтируется;
  * все суммы — в реальных ценах 2025 г. (инфляция исключена), r — реальная ставка;
  * приростный поток CF_t = (доход выпускника − прямые расходы) − доход сверстника без вуза;
  * срок окупаемости отсчитывается от окончания вуза (первый рабочий год выпускника = 1-й год);
  * IRR — корень NPV(r) = 0 на интервале (−50%; 100%).
"""
import numpy as np
from scipy import optimize

B1, B2 = 0.035, -0.0006      # профиль стажа (Минсер), общий для обеих траекторий
AGE0, RETIRE = 18, 64
T = RETIRE - AGE0 + 1         # 47 лет: t = 0..46
AGES = AGE0 + np.arange(T)

COUNTRY = {
    # w0 — стартовая месячная зарплата выпускника школы (цены 2025 г.)
    # k_g, k_s — отношение безработицы выпускников вузов / школ к общему уровню
    # m — возрастные множители безработицы: <25, 25–34, 35+
    "RUS": dict(w0=50_000, S=4, k_g=0.6, k_s=1.3, m=(3.25, 1.12, 0.85), compl=(0.68, 0.85)),
    "CHN": dict(w0=3_500, S=4, k_g=1.0, k_s=1.0, m=(3.4, 1.05, 0.8), compl=(0.90, 0.97)),
    # ЮАР: w0 откалиброван так, что медиана заработка занятых с полным средним образованием 18–64 лет
    # в модели равна 6283 рандам в мес. (Stats SA, Labour Market Dynamics 2024, табл. 3.22) в ценах 2025 г.
    # (ИПЦ 2025: +3,2%); k — отношения безработицы групп к общей во II кв. 2026 г. (12,4/33,6 и 36,0/33,6)
    "ZAF": dict(w0=4_140, S=3, k_g=0.369, k_s=1.071, m=(1.87, 1.22, 0.65), compl=(0.50, 0.72)),
}

# Шкалы налога на доходы физических лиц (годовой доход, нац. валюта, пороги считаются
# индексируемыми — заданы в ценах 2025–2026 гг.):
#   base = доход·(1 − ssc) − вычет;  налог = max(0, Σ_k (r_k − r_{k−1})·max(0, base − L_k) − rebate)
TAX = {
    # Россия: НДФЛ с 2025 г. (ст. 224 НК РФ): 13% до 2,4 млн, 15% до 5 млн, 18% до 20 млн, 20% до 50 млн, 22%
    "RUS": dict(L=[0, 2.4e6, 5e6, 20e6, 50e6], r=[0.13, 0.15, 0.18, 0.20, 0.22], ssc=0.0, ded=0.0, rebate=0.0),
    # Китай: ИНН по совокупному годовому доходу: вычет 60 000 юаней и взносы работника на соцстрахование ≈10,5%
    "CHN": dict(L=[0, 36e3, 144e3, 300e3, 420e3, 660e3, 960e3], r=[0.03, 0.10, 0.20, 0.25, 0.30, 0.35, 0.45],
                ssc=0.105, ded=60e3, rebate=0.0),
    # ЮАР: шкала SARS 2026/27 (налоговый год 1.03.2026–28.02.2027), основная скидка 17 820 рандов
    "ZAF": dict(L=[0, 245_100, 383_100, 530_200, 695_800, 887_000, 1_878_600],
                r=[0.18, 0.26, 0.31, 0.36, 0.39, 0.41, 0.45], ssc=0.0, ded=0.0, rebate=17_820.0),
}


def tax(iso, y):
    """Налог с годового дохода y (вектор), по шкале страны."""
    T = TAX[iso]
    base = np.maximum(0.0, np.asarray(y) * (1 - T["ssc"]) - T["ded"])
    r = np.array(T["r"])
    dr = np.diff(np.r_[0.0, r])
    t = sum(dr[k] * np.maximum(0.0, base - T["L"][k]) for k in range(len(r)))
    return np.maximum(0.0, t - T["rebate"])


def age_mult(m):
    return np.where(AGES < 25, m[0], np.where(AGES < 35, m[1], m[2]))


def macro_paths(g5, u5, u_longrun, g_lr=1.5):
    """Полные пути 2026–2072. g5, u5 — прогноз/симуляция на 2026–2030 (в %).
    После 2030 г.: рост линейно сходится к g_lr к 2045 г.; безработица — к u_longrun к 2040 г."""
    g = np.empty(T)
    g[:5] = g5
    g[5:20] = np.linspace(g5[-1], g_lr, 16)[1:]
    g[20:] = g_lr
    W = np.cumprod(1 + g / 100)          # индекс реальной зарплаты, 2025 = 1, W_0 = 1 + g_2026
    u = np.empty(T)
    u[:5] = u5
    u[5:15] = np.linspace(u5[-1], u_longrun, 11)[1:]
    u[15:] = u_longrun
    return W, u / 100


def income(P, W, u, prem_log, start, k, iso, on=None):
    """Ожидаемый годовой доход после налога траектории, начинающей работать в год `start`.
    Налог начисляется на заработок занятого (годовой), затем умножается на вероятность занятости."""
    on = on or {}
    t = np.arange(T)
    e = t - start
    prof = np.exp(B1 * e + B2 * e ** 2) if on.get("experience", True) else np.ones(T)
    Wx = W if on.get("growth", True) else np.ones(T)
    if on.get("unemployment", True):
        emp = 1 - np.clip(u * k * age_mult(P["m"]), 0, 0.95)
    else:
        emp = np.ones(T)
    gross = 12 * P["w0"] * Wx * np.exp(prem_log) * prof
    net = gross - (tax(iso, gross) if on.get("tax", True) else 0.0)
    return np.where(e >= 0, emp * net, 0.0)


def flows(iso, cost, prem_log, W, u, compl, on=None, P=None):
    """Возвращает (CF выпускника, CF абитуриента с учётом отчисления, доход без вуза)."""
    P = P or COUNTRY[iso]
    S = P["S"]
    t = np.arange(T)
    y_s = income(P, W, u, 0.0, 0, P["k_s"], iso, on)
    y_g = income(P, W, u, prem_log, S, P["k_g"], iso, on)
    d = int(np.ceil(S / 2))                       # отчисленный учится половину срока
    y_d = income(P, W, u, 0.0, d, P["k_s"], iso, on)
    grad = y_g - np.where(t < S, cost, 0.0) - y_s
    drop = y_d - np.where(t < d, cost, 0.0) - y_s
    exante = compl * grad + (1 - compl) * drop
    return grad, exante, y_s


def npv(cf, r):
    return float(np.sum(cf / (1 + r) ** np.arange(len(cf))))


def irr(cf):
    try:
        return optimize.brentq(lambda r: npv(cf, r), -0.5, 1.0, xtol=1e-10)
    except ValueError:
        return np.nan


def payback(cf, S, r=0.0):
    """Лет после окончания вуза до момента, когда накопленный (дисконтированный) поток ≥ 0.
    Внутри года — линейная интерполяция. NaN, если не окупается до 64 лет."""
    disc = cf / (1 + r) ** np.arange(len(cf))
    cum = np.cumsum(disc)
    idx = np.where((cum >= 0) & (np.arange(len(cf)) >= S))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    frac = -cum[i - 1] / disc[i]
    return (i - S) + frac


def sign_changes(cf):
    s = np.sign(cf[np.abs(cf) > 1e-9])
    return int(np.sum(s[1:] != s[:-1]))
