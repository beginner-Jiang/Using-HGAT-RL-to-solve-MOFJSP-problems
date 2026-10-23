# -*- coding: utf-8 -*-
"""Four clean layout variants of the inference-time scaling figure (Origin).

No free-floating text objects: their screen position does not survive the export reliably in
this build.  The series name carries the fit information and Origin places the legend, so the
layout cannot drift.  Each variant is exported as PNG into .../Variants.

Usage:  python PPO_HGAT/Time_Space_Complexity_Analysis/complexity_variants.py
"""
import csv
import os
import sys
import time

import numpy as np
import win32com.client

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Figure_And_File", "PPO_HGAT", "Time_Space_Complexity_Analysis")
VAR = os.path.join(OUT, "Variants")
SRC = os.path.join(OUT, "runtime_scaling_data.csv")
DPI = 300
BLUE = "color(46,111,186)"


def load():
    rows = []
    with open(SRC, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n = float(row["N"])
            total = float(row["time_ms"]) / 1000.0
            rows.append((row["scale"], n, total, float(row["time_ms"]) / n))
    n = np.array([row[1] for row in rows])
    t = np.array([row[2] for row in rows])
    slope, intercept = np.polyfit(np.log(n), np.log(t), 1)
    coefficient = float(np.exp(intercept))
    residual = ((np.log(t) - (intercept + slope * np.log(n))) ** 2).sum()
    total_ss = ((np.log(t) - np.log(t).mean()) ** 2).sum()
    r2 = float(1.0 - residual / total_ss)
    return rows, slope, coefficient, r2, t[-1] / t[0]


def write_csv(path, header, data):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def main():
    rows, slope, coefficient, r2, growth = load()
    os.makedirs(VAR, exist_ok=True)
    total_csv = os.path.join(VAR, "total.csv")
    step_csv = os.path.join(VAR, "step.csv")
    label = "Measured (fit T = %.2f N^%.2f, R^2 = %.3f)" % (coefficient, slope, r2)
    step_label = "Per step, 8.0 -> 18.7 ms (total x%.1f)" % growth
    write_csv(total_csv, ["N", label], [("%.1f" % r[1], "%.3f" % r[2]) for r in rows])
    write_csv(step_csv, ["N", step_label], [("%.1f" % r[1], "%.2f" % r[3]) for r in rows])

    app = win32com.client.Dispatch("Origin.ApplicationSI")
    app.Visible = 0
    time.sleep(2.0)

    def run(cmd, quiet=False):
        ok = app.Execute(cmd)
        time.sleep(0.3)
        if not ok and not quiet:
            print("FAILED: %s" % cmd)
        return ok

    def text(name):
        return app.LTStr(name)

    def build(csv_path, plot_type, xlog, ylog, xr, yr, xtitle, ytitle):
        run("newbook;")
        run('impasc fname:="%s";' % csv_path)
        book, sheet = text("page.name$"), text("wks.name$")
        run("win -t plot;")
        graph = text("%H")
        if not graph or not graph.startswith("Graph"):
            graph = "Graph1"
        run("win -a %s;" % book)
        run("plotxy iy:=[%s]%s!(1,2) plot:=%d ogl:=[%s]1!;" % (book, sheet, plot_type, graph))
        run("win -a %s;" % graph)
        run("page.active = 1;")
        run("layer.left = 16; layer.top = 14; layer.width = 80; layer.height = 74;")
        run("layer.x.type = %d;" % (1 if xlog else 0))
        run("layer.y.type = %d;" % (1 if ylog else 0))
        run("layer.x.from = %g; layer.x.to = %g;" % xr)
        run("layer.y.from = %g; layer.y.to = %g;" % yr)
        run('label -x "%s";' % xtitle)
        run('label -y "%s";' % ytitle)
        for item in ("layer.x.label", "layer.y.label"):
            run("%s.font = font(Times New Roman);" % item, quiet=True)
            run("%s.pts = 14;" % item, quiet=True)
        run("layer.plot1.color = %s;" % BLUE)
        return graph

    def legend():
        run("legend;")
        run("legend.font = font(Times New Roman);")
        run("legend.left = 3400; legend.top = 380;")

    def export(name):
        run('expGraph type:=png path:="%s" filename:="%s" overwrite:=replace '
            'keepsize:=1 tr.Advanced.DPI:=%d;' % (VAR, name, DPI), quiet=True)
        path = os.path.join(VAR, name + ".png")
        print("%-28s %s" % (name, "written" if os.path.exists(path) else "MISSING"))

    build(total_csv, 200, True, True, (50, 700), (0.35, 14),
          "Total number of operations N", "Inference time per schedule (s)")
    run("layer.plot1.line.width = 3;")
    legend()
    export("A_loglog_line")

    build(total_csv, 202, True, True, (50, 700), (0.35, 14),
          "Total number of operations N", "Inference time per schedule (s)")
    run("layer.plot1.line.width = 2;")
    run("layer.plot1.symbol.size = 7;")
    legend()
    export("B_loglog_line_markers")

    build(total_csv, 201, True, True, (50, 700), (0.35, 14),
          "Total number of operations N", "Inference time per schedule (s)")
    run("layer.plot1.symbol.size = 10;")
    legend()
    export("C_scatter")

    build(step_csv, 202, True, False, (50, 700), (6, 20),
          "Total number of operations N", "Inference time per decision step (ms)")
    run("layer.plot1.line.width = 2;")
    run("layer.plot1.symbol.size = 7;")
    legend()
    export("E_perstep_cost")

    app.Exit()
    print("variants written to", VAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())


