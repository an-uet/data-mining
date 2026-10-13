"""Table writing: CSV (full precision) + LaTeX (booktabs) with optional bold-best-in-column."""
import numpy as np
import pandas as pd

from . import config as C

_LATEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_LATEX_UNI = {"±": r"$\pm$", "−": r"$-$", "Δ": r"$\Delta$", "β": r"$\beta$", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "²": r"$^2$", "³": r"$^3$",
              "→": r"$\rightarrow$", "★": r"$\star$", "′": r"$'$", "∞": r"$\infty$", "ρ": r"$\rho$", "µ": r"$\mu$", "χ": r"$\chi$", "α": r"$\alpha$", "τ": r"$\tau$", "–": "--", "—": "---", "°": r"$^\circ$", "<": r"$<$", ">": r"$>$", "|": r"\textbar{}"}
TABLE_CAPTIONS = {}


def latex_escape(s):
    s = str(s).replace("->", "→").replace(">=", "≥").replace("<=", "≤")
    s = "".join(_LATEX_ESC.get(ch, ch) for ch in s)
    return "".join(_LATEX_UNI.get(ch, ch) for ch in s)


def df_to_latex(df, caption, label, float_fmt="{:.3f}", col_format=None, bold=None):
    """bold: set of (row position, column name) whose cell is printed in bold."""
    bold = bold or set()
    cols = list(df.columns)
    col_format = col_format or ("l" + "r" * (len(cols) - 1))

    def cell(v, i, c):
        if isinstance(v, (float, np.floating)):
            txt = "--" if not np.isfinite(v) else float_fmt.format(v)
        elif isinstance(v, (int, np.integer)):
            txt = f"{v:d}"
        else:
            txt = latex_escape(v)
        return rf"\textbf{{{txt}}}" if (i, c) in bold else txt

    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{col_format}}}", r"\toprule", " & ".join(latex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        lines.append(" & ".join(cell(v, i, c) for v, c in zip(row, cols)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def best_cells(num_df, rules, row_mask=None, decimals=None):
    """rules: {display column: (numeric column, 'min'|'max')} -> set of (row position, display column) holding the best value.
    decimals: {display column: number of printed decimals}; rows whose PRINTED value equals the best printed value are all marked (ties)."""
    out = set()
    for disp, (num, how) in rules.items():
        v = num_df[num].to_numpy(float).copy()
        if row_mask is not None:
            v = np.where(row_mask, v, np.nan)
        if np.isfinite(v).any():
            b = np.nanmin(v) if how == "min" else np.nanmax(v)
            d = (decimals or {}).get(disp)
            hit = np.isclose(np.round(v, d), np.round(b, d), rtol=0, atol=0) if d is not None else (v == b)
            out |= {(int(i), disp) for i in np.flatnonzero(hit & np.isfinite(v))}
    return out


def save_table(df, stem, caption, label=None, float_fmt="{:.3f}", col_format=None, csv_df=None, bold=None):
    """Write <stem>.csv (csv_df if given, else df) and <stem>.tex (needs \\usepackage{booktabs}). Returns df."""
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (df if csv_df is None else csv_df).to_csv(C.TABLES / f"{stem}.csv", index=False)
    (C.TABLES / f"{stem}.tex").write_text(df_to_latex(df, caption, label or f"tab:{stem}", float_fmt, col_format, bold))
    TABLE_CAPTIONS[stem] = caption
    return df
