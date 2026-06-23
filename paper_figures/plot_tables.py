"""Generate the FULL LaTeX tabulars of Table 3 (sampler comparison) and Table 4
(propagation effect) from plotdata.xlsx sheets `Table3-sampler`, `Table4-prop`,
and `Table34-stddev`. Writes tables/table3.tex and tables/table4.tex, each a
complete `\\begin{tabular}...\\end{tabular}`. 06_evaluation.tex wraps them with
`\\resizebox{\\columnwidth}{!}{\\input{tables/tableN.tex}}`.

NOTE: the whole tabular lives in the file (header + body + footer). Do NOT
\\input a mid-tabular fragment inside \\resizebox — that breaks booktabs rules
("Misplaced \\noalign"). Edit the sheets, rerun this script, recompile.

Run: python3 paper_figures/plot_tables.py
"""
from pathlib import Path
import statistics as st
import openpyxl

HERE = Path(__file__).resolve().parent
XLSX = HERE / "plotdata.xlsx"
OUT = HERE / "tables"

COLSPEC = r"\begin{tabular}{@{}l rr rr rr rr@{}}"
HDR_QUERY = r"Query & $F_1$ & Lat. & $F_1$ & Lat. & $F_1$ & Lat. & $F_1$ & Lat. \\"
CMID = r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}"
HDR3 = (r"& \multicolumn{2}{c}{uniform$+$LR} & \multicolumn{2}{c}{topk$+$LR} & "
        r"\multicolumn{2}{c}{active(UQE)$+$LR} & \multicolumn{2}{c}{active(LASO)$+$LR} \\")
HDR4 = (r"& \multicolumn{2}{c}{uniform$+$LR} & \multicolumn{2}{c}{uniform$+$LP} & "
        r"\multicolumn{2}{c}{active(LASO)$+$LR} & \multicolumn{2}{c}{active(LASO)$+$LP} \\")


# Low-selectivity queries moved to the case study; dropped from the main
# 14-query workload tables. Keep names in sheet-native form (see Table*-sampler).
SKIP_QUERIES = {"ecomm_q6", "movie_q5"}


def load(sheet):
    ws = openpyxl.load_workbook(XLSX, data_only=True)[sheet]
    return [r for r in ws.iter_rows(min_row=2, values_only=True)
            if r[0] is not None and str(r[0]) not in SKIP_QUERIES]


def f1(x):
    return f"{x:.2f}"


def lat(x):
    return f"{x:.1f}"


def body(rows, stddev, delta_kind):
    out = []
    for r in rows:
        q = str(r[0])
        bc = int(r[9])
        cells = []
        for n in range(4):
            v_f1, v_lat = r[1 + 2 * n], r[2 + 2 * n]
            s = (f"$\\mathbf{{{f1(v_f1)}}}$" if bc == n + 1 else f"${f1(v_f1)}$")
            cells += [s, f"${lat(v_lat)}$"]
        qtex = "\\texttt{" + q.replace("_", "\\_") + "}"
        out.append(f"{qtex:<20} & " + " & ".join(cells) + r" \\")
    fav = [round(st.mean([r[1 + 2 * n] for r in rows]), 2) for n in range(4)]
    lav = [round(st.mean([r[2 + 2 * n] for r in rows]), 1) for n in range(4)]
    out.append(r"\midrule")
    acells = []
    for n in range(4):
        acells += [f"$\\mathbf{{{f1(fav[n])}}}$" if n == 3 else f"${f1(fav[n])}$",
                   f"${lav[n]}$"]
    out.append(r"\textbf{Average}      & " + " & ".join(acells) + r" \\")
    sd = stddev
    scells = []
    for n in range(4):
        scells += [f"$\\pm{sd[1 + 2 * n]:g}$", f"$\\pm{sd[2 + 2 * n]:g}$"]
    out.append(r"\textbf{Std Dev}      & " + " & ".join(scells) + r" \\")
    out.append(r"\midrule")
    if delta_kind == "LASO":
        d = []
        for n in range(3):
            d += [f"${fav[3] - fav[n]:+.2f}$", f"${lav[3] / lav[n]:.2f}" + r"\times$"]
        d += ["---", "---"]
        out.append(r"\textbf{$\Delta$\,LASO} & " + " & ".join(d) + r" \\")
    else:
        du, fu = fav[1] - fav[0], lav[1] / lav[0]
        dl, fl = fav[3] - fav[2], lav[3] / lav[2]
        out.append(r"\textbf{$\Delta$\,LP} & --- & --- & "
                   f"${du:+.2f}$ & ${fu:.2f}" + r"\times$ & --- & --- & "
                   f"${dl:+.2f}$ & ${fl:.2f}" + r"\times$ \\")
    return "\n".join(out)


def table(header, rows, stddev, kind):
    return "\n".join([COLSPEC, r"\toprule", header, CMID, HDR_QUERY, r"\midrule",
                      body(rows, stddev, kind), r"\bottomrule", r"\end{tabular}"]) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    sd = {r[0]: r for r in load("Table34-stddev")}
    (OUT / "table3.tex").write_text(table(HDR3, load("Table3-sampler"), sd["Table3"], "LASO"))
    (OUT / "table4.tex").write_text(table(HDR4, load("Table4-prop"), sd["Table4"], "LP"))
    print("wrote tables/table3.tex and tables/table4.tex (full tabulars)")


if __name__ == "__main__":
    main()
