"""Excel-сверка: независимая реализация центрального расчёта формулами Excel."""
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as CL
from openpyxl.comments import Comment

BASE = Path(__file__).resolve().parent
R = json.loads((BASE / "out_v3" / "results.json").read_text())
BR = json.loads((BASE / "out_v3" / "bridge.json").read_text())
OUT = BASE / "out_v3" / "Сверка_расчётов_окупаемость_v3.xlsx"

F = "Times New Roman"
BLUE = Font(name=F, color="0000FF")
BLK = Font(name=F)
GRN = Font(name=F, color="008000")
BOLD = Font(name=F, bold=True)
HDR = PatternFill("solid", fgColor="DDE4F0")
YEL = PatternFill("solid", fgColor="FFF2B3")
thin = Side(style="thin", color="999999")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="center")

SC = [("RUS_budget", "РФ_бюджет"), ("RUS_paid_avg", "РФ_платно_ср"), ("RUS_paid_top", "РФ_платно_топ"),
      ("CHN_public", "КНР_гос"), ("ZAF_self", "ЮАР_семья"), ("ZAF_nsfas", "ЮАР_NSFAS")]
CTRY = R["params"]["country"]
NAMES = {"RUS": "Россия", "CHN": "Китай", "ZAF": "ЮАР"}

wb = Workbook()
# ---------------------------------------------------------------- Описание
ws = wb.active
ws.title = "Описание"
lines = [
    ("Сверка расчётов окупаемости высшего образования (центральный сценарий, модель v3)", BOLD),
    ("", BLK),
    ("Что это. Независимая реализация модели жизненного цикла формулами Excel. Каждый лист сценария "
     "строит по годам (t = 0 — первый год обучения, 2026 г., возраст 18 лет; последний — 64 года) доходы "
     "человека без вуза, выпускника и отчисленного, приростной денежный поток и показатели NPV, IRR, сроки "
     "окупаемости. Лист «Сверка» сравнивает результаты Excel с расчётом на Python.", BLK),
    ("Входные данные — лист «Параметры» (синий шрифт = вводимые значения, жёлтая заливка — переключатели). "
     "Меняя их, можно пересчитать любой сценарий. Переключатели 1/0 включают или выключают стаж, налоги, "
     "безработицу и рост зарплат — так воспроизводится разложение «статика → динамика» (лист «Мост»).", BLK),
    ("Центральный сценарий: прогнозы ARIMA (точечные) на 2026–2030 гг., премия — медиана (Россия) или мода "
     "(Китай, ЮАР) распределения, доля окончивших — середина диапазона. Табл. 2 статьи содержит медианы "
     "по 10 000 симуляций Монте-Карло, поэтому значения близки, но не обязаны совпадать (медиана IRR "
     "≠ IRR при медианных параметрах).", BLK),
    ("Соглашения: потоки относятся к началу года, множитель дисконтирования (1 + r)^(−t); NPV = CF0 + "
     "NPV(r; CF1:CF46) в обозначениях Excel. Все суммы — в реальных ценах 2025 г., национальная валюта.", BLK),
    ("Срок окупаемости — число лет после окончания вуза до момента, когда накопленный поток выпускника "
     "становится ≥ 0 (с линейной интерполяцией внутри года); дисконтированный — то же для дисконтированного потока.", BLK),
]
for i, (t, f) in enumerate(lines, start=1):
    c = ws.cell(row=i, column=1, value=t)
    c.font = f
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 120
for i in range(3, 8):
    ws.row_dimensions[i].height = 48

# ---------------------------------------------------------------- Параметры
P = wb.create_sheet("Параметры")
P["A1"] = "Параметр"
P["A1"].font = BOLD
rows = ["Страна", "w0 — стартовая зарплата выпускника школы, в месяц", "S — срок обучения, лет",
        "Прямые расходы на обучение в год", "Налог: шкала страны (лист «Налоги»)", "—",
        "k — множитель безработицы выпускника вуза", "k — множитель безработицы без вуза",
        "m — возрастной множитель, <25 лет", "m — 25–34 года", "m — 35+ лет",
        "β2 — стаж", "β3 — стаж²", "π — логарифм премии", "p — доля окончивших вуз",
        "r — реальная ставка дисконтирования", "d — лет учёбы отчисленного = ОКРВВЕРХ(S/2)",
        "g 2026, % (ARIMA)", "g 2027, %", "g 2028, %", "g 2029, %", "g 2030, %",
        "u 2026, % (ARIMA)", "u 2027, %", "u 2028, %", "u 2029, %", "u 2030, %",
        "u долгосрочная (среднее 2010–2025), %", "g долгосрочный (с 2045 г.), %",
        "Переключатель: стаж (1/0)", "Переключатель: налоги (1/0)",
        "Переключатель: безработица (1/0)", "Переключатель: рост зарплат (1/0)", "Премия, % = exp(π) − 1"]
for i, r in enumerate(rows, start=2):
    P.cell(row=i, column=1, value=r).font = BLK
ROW = {name: i for i, name in enumerate(["country", "w0", "S", "cost", "tax_g", "tax_s", "k_g", "k_s",
                                         "m1", "m2", "m3", "b2", "b3", "prem", "p", "r", "d",
                                         "g0", "g1", "g2", "g3", "g4", "u0", "u1", "u2", "u3", "u4",
                                         "ulr", "glr", "sw_exp", "sw_tax", "sw_un", "sw_gr", "prem_pct"], start=2)}
for j, (sc, sh) in enumerate(SC, start=2):
    col = CL(j)
    c = R["central"][sc]
    cc = CTRY[c["iso"]]
    P.cell(row=1, column=j, value=sh).font = BOLD
    vals = {"country": NAMES[c["iso"]], "w0": cc["w0"], "S": cc["S"], "cost": c["cost"],
            "tax_g": {"RUS": "НДФЛ РФ", "CHN": "ИНН КНР", "ZAF": "SARS 2026/27"}[c["iso"]], "tax_s": "", "k_g": cc["k_g"], "k_s": cc["k_s"],
            "m1": cc["m"][0], "m2": cc["m"][1], "m3": cc["m"][2], "b2": R["params"]["B1"],
            "b3": R["params"]["B2"], "prem": c["prem_log"], "p": c["compl"], "r": R["params"]["R"],
            "ulr": c["u_longrun"], "glr": 1.5, "sw_exp": 1, "sw_tax": 1, "sw_un": 1, "sw_gr": 1}
    for k in range(5):
        vals[f"g{k}"] = c["g5"][k]
        vals[f"u{k}"] = c["u5"][k]
    for k, v in vals.items():
        cell = P[f"{col}{ROW[k]}"]
        cell.value = v
        cell.font = BLUE
        if k.startswith("sw_"):
            cell.fill = YEL
    P[f"{col}{ROW['d']}"] = f"=ROUNDUP({col}{ROW['S']}/2,0)"
    P[f"{col}{ROW['prem_pct']}"] = f"=EXP({col}{ROW['prem']})-1"
    for k in ["p", "r", "prem_pct"]:
        P[f"{col}{ROW[k]}"].number_format = "0.0%"
    P[f"{col}{ROW['prem']}"].number_format = "0.0000"
    for k in ["w0", "cost"]:
        P[f"{col}{ROW[k]}"].number_format = "#,##0"
P.column_dimensions["A"].width = 52
for j in range(2, 8):
    P.column_dimensions[CL(j)].width = 15
src = {"w0": "Россия ≈ ½ средней зарплаты 2025 г. (100 360 руб., Росстат); Китай — раздел по Китаю; "
             "ЮАР — медиана matric 5 700 рандов (Stats SA, 2022) с поправкой на стаж",
       "cost": "РФ: 270 тыс. ≈ 2,7 среднемесячной зарплаты (мониторинг НИУ ВШЭ 2025), 450 тыс. — ведущий вуз; "
               "КНР, ЮАР — разделы по странам",
       "prem": "РФ — регрессия премии на охват (медиана прогноза 2030 г.); КНР — мода 45%; ЮАР — 23%×3 года (Köhler 2025)",
       "p": "РФ: отсев 15–32% (Минобрнауки, 2025); КНР: >90% (Sixth Tone); ЮАР: ~50% (Van Broekhuizen et al., 2016)",
       "k_g": "ЮАР: 12,2/33,2 и 35,2/33,2 (QLFS II кв. 2025); РФ, КНР — допущение",
       "m1": "РФ: 10,4/3,2 (20–24 года, 2023); КНР: ~17/5; ЮАР: 62,2/33,2 (QLFS)",
       "g0": "ARIMA, точечный прогноз (блок 1)", "u0": "ARIMA, точечный прогноз (блок 1), модельная оценка МОТ",
       "tax_g": "Прогрессивные шкалы: НДФЛ (ст. 224 НК РФ, с 2025 г.); ИНН КНР (вычет 60 тыс. юаней, взносы 10,5%); SARS 2026/27 (скидка 17 820)"}
P.cell(row=1, column=8, value="Источник / комментарий").font = BOLD
for k, t in src.items():
    c = P.cell(row=ROW[k], column=8, value=t)
    c.font = Font(name=F, italic=True, size=9)
P.column_dimensions["H"].width = 90
for row in P.iter_rows(min_row=1, max_row=ROW["prem_pct"], min_col=1, max_col=7):
    for c in row:
        c.border = BOX
for j in range(1, 9):
    P.cell(row=1, column=j).fill = HDR

# ---------------------------------------------------------------- Налоги
import lifecycle3 as L3
TX = wb.create_sheet("Налоги")
TX["A1"] = "Шкалы налога на доходы (годовой доход, нац. валюта): налог = max(0; Σ Δr·max(0; база − L) − скидка), база = max(0; доход·(1 − взносы) − вычет)"
TX["A1"].font = BOLD
TAXROW = {}
r0 = 3
for iso, nm in [("RUS", "Россия"), ("CHN", "Китай"), ("ZAF", "ЮАР")]:
    T = L3.TAX[iso]
    TX.cell(row=r0, column=1, value=nm).font = BOLD
    TX.cell(row=r0 + 1, column=1, value="Нижняя граница ступени L")
    TX.cell(row=r0 + 2, column=1, value="Предельная ставка r")
    TX.cell(row=r0 + 3, column=1, value="Прирост ставки Δr")
    for k, (lo, rt) in enumerate(zip(T["L"], T["r"])):
        col = CL(2 + k)
        TX[f"{col}{r0 + 1}"] = lo
        TX[f"{col}{r0 + 1}"].font = BLUE
        TX[f"{col}{r0 + 1}"].number_format = "#,##0"
        TX[f"{col}{r0 + 2}"] = rt
        TX[f"{col}{r0 + 2}"].font = BLUE
        TX[f"{col}{r0 + 2}"].number_format = "0%"
        TX[f"{col}{r0 + 3}"] = f"={col}{r0 + 2}" if k == 0 else f"={col}{r0 + 2}-{CL(1 + k)}{r0 + 2}"
        TX[f"{col}{r0 + 3}"].number_format = "0%"
    last = CL(1 + len(T["L"]))
    for j, (lab, v) in enumerate([("Взносы работника (доля)", T["ssc"]), ("Вычет", T["ded"]), ("Скидка (rebate)", T["rebate"])]):
        TX.cell(row=r0 + 1 + j, column=11, value=lab)
        TX.cell(row=r0 + 1 + j, column=12, value=v).font = BLUE
    TAXROW[iso] = dict(L=f"Налоги!$B${r0 + 1}:${last}${r0 + 1}", dR=f"Налоги!$B${r0 + 3}:${last}${r0 + 3}",
                       ssc=f"Налоги!$L${r0 + 1}", ded=f"Налоги!$L${r0 + 2}", reb=f"Налоги!$L${r0 + 3}")
    r0 += 6
TX.column_dimensions["A"].width = 28
TX.column_dimensions["K"].width = 26
for k in range(2, 10):
    TX.column_dimensions[CL(k)].width = 13

# ---------------------------------------------------------------- Листы сценариев
HEAD = ["t", "Год", "Возраст", "g_t, %", "W_t (индекс зарплат)", "u_t (доля)", "m (возраст)",
        "Стаж без вуза", "Стаж выпускника", "Стаж отчисленного",
        "Доход без вуза", "Доход выпускника", "Доход отчисленного",
        "Расходы выпускника", "Расходы отчисленного",
        "CF выпускника", "CF отчисленного", "CF абитуриента", "Множитель дисконт.",
        "Накопл. CF выпускника", "Накопл. дисконт. CF выпускника",
        "Флаг простой окупаемости", "Флаг дисконт. окупаемости",
        "Заработок занятого без вуза (до налога)", "То же, выпускник", "То же, отчисленный",
        "Налог без вуза", "Налог выпускника", "Налог отчисленного"]
N0, N1 = 4, 50          # строки t = 0..46
CHECK = {}
for j, (sc, sh) in enumerate(SC, start=2):
    pc = CL(j)                        # колонка сценария на листе «Параметры»
    p = lambda k: f"Параметры!${pc}${ROW[k]}"
    ws = wb.create_sheet(sh)
    ws["A1"] = f"Сценарий {sh}: {R['mc'][sc]['label']} — центральный расчёт по годам"
    ws["A1"].font = BOLD
    for k, h in enumerate(HEAD, start=1):
        c = ws.cell(row=3, column=k, value=h)
        c.font = BOLD
        c.fill = HDR
        c.alignment = WRAP
        c.border = BOX
    ws.row_dimensions[3].height = 45
    for t in range(47):
        r = N0 + t
        prev = r - 1
        ws[f"A{r}"] = t
        ws[f"B{r}"] = f"=2026+A{r}"
        ws[f"C{r}"] = f"=18+A{r}"
        # g_t
        if t < 5:
            ws[f"D{r}"] = f"={p('g' + str(t))}"
        else:
            ws[f"D{r}"] = (f"=IF(A{r}>=20,{p('glr')},{p('g4')}+({p('glr')}-{p('g4')})*(A{r}-4)/15)")
        # W_t (рост можно выключить)
        ws[f"E{r}"] = (f"=IF({p('sw_gr')}=1,(1+D{r}/100),1)" if t == 0 else
                       f"=IF({p('sw_gr')}=1,E{prev}*(1+D{r}/100),1)")
        # u_t
        if t < 5:
            ws[f"F{r}"] = f"={p('u' + str(t))}/100"
        else:
            ws[f"F{r}"] = (f"=IF(A{r}>=15,{p('ulr')},{p('u4')}+({p('ulr')}-{p('u4')})*(A{r}-4)/10)/100")
        ws[f"G{r}"] = f"=IF(C{r}<25,{p('m1')},IF(C{r}<35,{p('m2')},{p('m3')}))"
        ws[f"H{r}"] = f"=A{r}"
        ws[f"I{r}"] = f"=A{r}-{p('S')}"
        ws[f"J{r}"] = f"=A{r}-{p('d')}"

        TXR = TAXROW[R["central"][sc]["iso"]]
        for gcol, ecol, prem in [("X", "H", "0"), ("Y", "I", p("prem")), ("Z", "J", "0")]:
            prof = f"IF({p('sw_exp')}=1,EXP({p('b2')}*{ecol}{r}+{p('b3')}*{ecol}{r}^2),1)"
            ws[f"{gcol}{r}"] = f"=12*{p('w0')}*E{r}*EXP({prem})*{prof}"
        for tcol, gcol in [("AA", "X"), ("AB", "Y"), ("AC", "Z")]:
            base = f"MAX(0,{gcol}{r}*(1-{TXR['ssc']})-{TXR['ded']})"
            ws[f"{tcol}{r}"] = (f"=MAX(0,SUMPRODUCT(({base}>{TXR['L']})*({base}-{TXR['L']})*{TXR['dR']})-{TXR['reb']})"
                                f"*{p('sw_tax')}")

        def inc(ecol, gcol, tcol, kk):
            emp = f"IF({p('sw_un')}=1,1-MIN(MAX(F{r}*{p(kk)}*G{r},0),0.95),1)"
            return f"=IF({ecol}{r}>=0,{emp}*({gcol}{r}-{tcol}{r}),0)"
        ws[f"K{r}"] = inc("H", "X", "AA", "k_s")
        ws[f"L{r}"] = inc("I", "Y", "AB", "k_g")
        ws[f"M{r}"] = inc("J", "Z", "AC", "k_s")
        for col in ["X", "Y", "Z", "AA", "AB", "AC"]:
            ws[f"{col}{r}"].number_format = "#,##0"
        ws[f"N{r}"] = f"=IF(A{r}<{p('S')},{p('cost')},0)"
        ws[f"O{r}"] = f"=IF(A{r}<{p('d')},{p('cost')},0)"
        ws[f"P{r}"] = f"=L{r}-N{r}-K{r}"
        ws[f"Q{r}"] = f"=M{r}-O{r}-K{r}"
        ws[f"R{r}"] = f"={p('p')}*P{r}+(1-{p('p')})*Q{r}"
        ws[f"S{r}"] = f"=1/(1+{p('r')})^A{r}"
        ws[f"T{r}"] = f"=P{r}" if t == 0 else f"=T{prev}+P{r}"
        ws[f"U{r}"] = f"=P{r}*S{r}" if t == 0 else f"=U{prev}+P{r}*S{r}"
        if t == 0:
            ws[f"V{r}"] = '=""'
            ws[f"W{r}"] = '=""'
        else:
            ws[f"V{r}"] = (f"=IF(AND(A{r}>={p('S')},T{r}>=0,T{prev}<0),(A{r}-{p('S')})+(-T{prev}/P{r}),\"\")")
            ws[f"W{r}"] = (f"=IF(AND(A{r}>={p('S')},U{r}>=0,U{prev}<0),(A{r}-{p('S')})+(-U{prev}/(P{r}*S{r})),\"\")")
        for col in "KLMNOPQRTU":
            ws[f"{col}{r}"].number_format = "#,##0;-#,##0"
        ws[f"E{r}"].number_format = "0.0000"
        ws[f"F{r}"].number_format = "0.00%"
        ws[f"S{r}"].number_format = "0.0000"
        for col in "VW":
            ws[f"{col}{r}"].number_format = "0.00"
    # итоговый блок
    rng = lambda col: f"{col}{N0}:{col}{N1}"
    summ = [("NPV выпускника (r) = SUMPRODUCT", f"=SUMPRODUCT({rng('P')},{rng('S')})", "#,##0"),
            ("NPV выпускника — функция NPV Excel", f"=P{N0}+NPV({p('r')},P{N0+1}:P{N1})", "#,##0"),
            ("NPV абитуриента (с учётом отчисления)", f"=R{N0}+NPV({p('r')},R{N0+1}:R{N1})", "#,##0"),
            ("NPV абитуриента, в годовых заработках выпускника школы", f"=AF6/(12*{p('w0')})", "0.00"),
            ("IRR выпускника", f"=IRR({rng('P')},0.1)", "0.00%"),
            ("IRR абитуриента", f"=IRR({rng('R')},0.1)", "0.00%"),
            ("Простой срок окупаемости, лет после выпуска", f"=MIN({rng('V')})", "0.00"),
            ("Дисконтированный срок окупаемости, лет", f"=MIN({rng('W')})", "0.00"),
            ("ROI выпускника за 10 лет после выпуска, недисконт.",
             f"=SUMPRODUCT((A{N0}:A{N1}>={p('S')})*(A{N0}:A{N1}<{p('S')}+10)*P{N0}:P{N1})"
             f"/-SUMPRODUCT((A{N0}:A{N1}<{p('S')})*P{N0}:P{N1})-1", "0.0%")]
    ws["AE3"], ws["AF3"] = "Показатель", "Значение (Excel)"
    ws["AE3"].font = ws["AF3"].font = BOLD
    ws["AE3"].fill = ws["AF3"].fill = HDR
    for k, (lab, fml, fmt) in enumerate(summ, start=4):
        ws[f"AE{k}"] = lab
        ws[f"AF{k}"] = fml
        ws[f"AF{k}"].number_format = fmt
        ws[f"AE{k}"].border = ws[f"AF{k}"].border = BOX
    CHECK[sc] = dict(sheet=sh, npv_g="AF4", npv_g2="AF5", npv_x="AF6", irr_g="AF8", irr_x="AF9", pb_g="AF10", pbd_g="AF11")
    ws.column_dimensions["AE"].width = 52
    ws.column_dimensions["AF"].width = 18
    for k in range(1, 30):
        ws.column_dimensions[CL(k)].width = 13
    ws.freeze_panes = "D4"

# ---------------------------------------------------------------- Сверка
S = wb.create_sheet("Сверка", 1)
S["A1"] = "Сверка: Excel (формулы) vs Python (lifecycle3.py), центральный расчёт; справа — табл. 2 (Монте-Карло)"
S["A1"].font = BOLD
hdr = ["Сценарий", "Показатель", "Excel", "Python", "|Разница|", "Совпадает (допуск)",
       "Табл. 2: медиана МК", "Табл. 2: 10-й проц.", "Табл. 2: 90-й проц."]
for k, h in enumerate(hdr, start=1):
    c = S.cell(row=3, column=k, value=h)
    c.font = BOLD
    c.fill = HDR
    c.alignment = WRAP
    c.border = BOX
METR = [("npv_g", "NPV выпускника, нац. валюта", "npv_g", 1.0, None),
        ("npv_g2", "NPV выпускника (функция NPV)", "npv_g", 1.0, None),
        ("npv_x", "NPV абитуриента, нац. валюта", "npv_x", 1.0, None),
        ("irr_g", "IRR выпускника", "irr_g", 1e-6, "irr_grad"),
        ("irr_x", "IRR абитуриента", "irr_x", 1e-6, "irr_exante"),
        ("pb_g", "Простой срок окупаемости, лет", "pb_g", 1e-6, "payback_grad"),
        ("pbd_g", "Дисконт. срок окупаемости, лет", "pbd_g", 1e-6, "payback_disc_grad")]
r = 4
for sc, sh in SC:
    for key, lab, pykey, tol, mckey in METR:
        S.cell(row=r, column=1, value=sh)
        S.cell(row=r, column=2, value=lab)
        c = S.cell(row=r, column=3, value=f"={sh}!{CHECK[sc][key]}")
        c.font = GRN
        v = R["central"][sc][pykey]
        d = S.cell(row=r, column=4, value=v)
        d.font = BLUE
        d.comment = Comment("Значение из out_v3/results.json (central), рассчитано lifecycle3.py", "model_v3")
        S.cell(row=r, column=5, value=f"=ABS(C{r}-D{r})")
        tol_cell = tol if "npv" not in key else f"MAX(1,ABS(D{r})*1E-9)"
        S.cell(row=r, column=6, value=f"=IF(E{r}<={tol_cell},\"да\",\"НЕТ\")")
        if mckey:
            m = R["mc"][sc][mckey]
            for k, q in enumerate(["median", "p10", "p90"], start=7):
                S.cell(row=r, column=k, value=m[q]).font = BLUE
        fmt = "0.00%" if "irr" in key else ("0.00" if "pb" in key else "#,##0")
        for k in [3, 4, 7, 8, 9]:
            S.cell(row=r, column=k).number_format = fmt
        S.cell(row=r, column=5).number_format = "0.0E+00"
        for k in range(1, 10):
            S.cell(row=r, column=k).border = BOX
        r += 1
S.cell(row=r + 1, column=1, value="Итог сверки:").font = BOLD
S.cell(row=r + 1, column=2, value=f'=IF(COUNTIF(F4:F{r-1},"НЕТ")=0,"все показатели совпадают","есть расхождения")').font = BOLD
for k, w in enumerate([16, 34, 16, 16, 12, 14, 14, 14, 14], start=1):
    S.column_dimensions[CL(k)].width = w

# ---------------------------------------------------------------- Мост
M = wb.create_sheet("Мост", 2)
M["A1"] = "Почему статический и динамический сроки окупаемости различаются (центральный расчёт, лет после выпуска)"
M["A1"].font = BOLD
M["A2"] = ("Шаги добавляются последовательно; каждый можно воспроизвести переключателями на листе «Параметры». "
           "Шаг B совпадает с формулой PP = C/ΔY на входах модели — проверка согласованности.")
M["A2"].font = Font(name=F, italic=True)
labels = {"RUS_budget": "РФ, бюджет", "RUS_paid_top": "РФ, платно (450 тыс.)", "CHN_public": "Китай", "ZAF_self": "ЮАР"}
order = ["RUS_budget", "RUS_paid_top", "CHN_public", "ZAF_self"]
M.cell(row=4, column=1, value="Шаг").font = BOLD
M.cell(row=4, column=2, value="Что добавляется").font = BOLD
for k, sc in enumerate(order, start=3):
    M.cell(row=4, column=k, value=labels[sc]).font = BOLD
steps = ["A", "B", "C", "D", "E", "F"]
names = {}
vals = {}
for sc in order:
    for code, name, pb, irr in BR[sc]["rows"]:
        names[code] = name
        vals[(code, sc)] = pb
for i, code in enumerate(steps, start=5):
    M.cell(row=i, column=1, value=code)
    M.cell(row=i, column=2, value=names[code])
    for k, sc in enumerate(order, start=3):
        v = vals.get((code, sc))
        c = M.cell(row=i, column=k, value=v if v is not None else "—")
        c.number_format = "0.0"
        c.font = BLUE
M.cell(row=11, column=1, value="G")
M.cell(row=11, column=2, value="Медиана Монте-Карло (табл. 2)")
for k, sc in enumerate(order, start=3):
    c = M.cell(row=11, column=k, value=BR[sc]["mc_median"])
    c.number_format = "0.0"
    c.font = BLUE
for row in M.iter_rows(min_row=4, max_row=11, min_col=1, max_col=6):
    for c in row:
        c.border = BOX
        if c.row == 4:
            c.fill = HDR
M.column_dimensions["A"].width = 6
M.column_dimensions["B"].width = 48
for k in range(3, 7):
    M.column_dimensions[CL(k)].width = 20

for wsx in wb.worksheets:
    for row in wsx.iter_rows():
        for c in row:
            if c.font and c.font.name != F:
                c.font = Font(name=F, bold=c.font.bold, italic=c.font.italic, color=c.font.color, size=c.font.size)
wb.save(OUT)
print("saved", OUT)
