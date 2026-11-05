"""Пошаговое разложение: от статического срока окупаемости к динамическому (центральный расчёт v2)."""
import json, numpy as np
import lifecycle3 as L
R = json.load(open('out_v3/results.json'))
TEAM = {"RUS_paid_top": (2.5e6, 300e3), "CHN_public": (250e3, 27e3), "ZAF_self": (450e3, 96e3), "RUS_budget": None}
STEPS = [("B", "Статическая формула на входах модели", dict(experience=False, tax=False, unemployment=False, growth=False)),
         ("C", "+ потерянный стаж (профиль Минсера)", dict(experience=True, tax=False, unemployment=False, growth=False)),
         ("D", "+ налоги", dict(experience=True, tax=True, unemployment=False, growth=False)),
         ("E", "+ безработица (возраст × образование)", dict(experience=True, tax=True, unemployment=True, growth=False)),
         ("F", "+ рост реальных зарплат (полная модель)", dict(experience=True, tax=True, unemployment=True, growth=True))]
out = {}
for sc in ["RUS_budget", "RUS_paid_top", "CHN_public", "ZAF_self"]:
    c = R['central'][sc]; iso = c['iso']; P = L.COUNTRY[iso]
    W = np.array(c['W']); u = np.array(c['u'])
    rows = []
    if TEAM[sc]:
        C0, dY = TEAM[sc]; rows.append(("A", "Статический расчёт из раздела по стране", C0 / dY, None))
    # аналитическая проверка шага B
    Cm = P['S'] * (c['cost'] + 12 * P['w0']); dYm = 12 * P['w0'] * (np.exp(c['prem_log']) - 1)
    for code, name, on in STEPS:
        g, x, _ = L.flows(iso, c['cost'], c['prem_log'], W, u, c['compl'], on=on)
        rows.append((code, name, L.payback(g, P['S']), L.irr(g)))
    out[sc] = dict(rows=rows, analytic_B=Cm / dYm, C_model=Cm, dY_model=dYm,
                   mc_median=R['mc'][sc]['payback_grad']['median'])
    print(sc, 'analytic B =', round(Cm / dYm, 3), '| C =', round(Cm), 'dY =', round(dYm))
    for r in rows:
        print('   ', r[0], r[1], round(r[2], 2), '' if r[3] is None else f'IRR {r[3]*100:.1f}%')
    print('    MC median (табл. 2):', round(out[sc]['mc_median'], 2))
json.dump(out, open('out_v3/bridge.json', 'w'), ensure_ascii=False, indent=1, default=float)
