"""Generates notebooks/reproduce_results.ipynb (run from the repository root)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Reproduce every reported number

<a href="https://colab.research.google.com/github/badrelkhalyly-etu-alt/moodle-nonparticipation-xai/blob/main/notebooks/reproduce_results.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook re-runs the complete pipeline (`src/analysis.py`) on the raw data and compares **each numeric value** it produces with the reference output stored in the repository (`results/results.json`), from which all numbers in the manuscript are taken.

1. Run the cells in order. In Colab you will be asked to upload the data files (they are not distributed, see `data/README.md`):
   - `udk_moodle_all_grades.csv`
   - the four `logs_*.csv` event logs
2. The full analysis takes about 15 minutes on a standard Colab CPU.
3. The last cells list every value with its status: `MATCH` (identical up to 1e-6) or `MISMATCH`.

Pinned library versions are installed first, because tree ensembles and cross-validation splits can change slightly across library versions."""),
    code("""# 1. Get the code (Colab) and install the pinned versions
import os, sys, subprocess
REPO_URL = "https://github.com/badrelkhalyly-etu-alt/moodle-nonparticipation-xai.git"
if not os.path.exists("src/analysis.py") and os.path.exists("../src/analysis.py"):
    os.chdir("..")                      # notebook opened from a local clone (notebooks/ folder)
if not os.path.exists("src/analysis.py"):
    if not os.path.exists("moodle-nonparticipation-xai"):
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir("moodle-nonparticipation-xai")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print(os.getcwd())"""),
    code("""# 2. Data: upload the CSV files into data/ if they are not there yet
import glob, shutil
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/udk_moodle_all_grades.csv") or len(glob.glob("data/logs_*.csv")) < 4:
    if "google.colab" in sys.modules:
        from google.colab import files
        print("Select udk_moodle_all_grades.csv and the four logs_*.csv files")
        for name in files.upload():
            shutil.move(name, os.path.join("data", name))
    else:
        raise FileNotFoundError("Place udk_moodle_all_grades.csv and logs_*.csv in data/")
print(sorted(os.listdir("data")))"""),
    md("## 3. Re-run the complete analysis\nWrites `results/recomputed.json` and the figures; progress is printed table by table."),
    code("""FAST = False   # True = reduced grid and 200 bootstrap resamples (smoke test; values will not match)
cmd = [sys.executable, "-u", "src/analysis.py", "--data-dir", "data", "--out", "results/recomputed.json"]
if FAST:
    cmd.append("--fast")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
assert proc.wait() == 0, "analysis failed"
"""),
    md("## 4. Compare every value with the reference output"),
    code("""import json, math
import pandas as pd

ref = json.load(open("results/results.json", encoding="utf-8"))
new = json.load(open("results/recomputed.json", encoding="utf-8"))

def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj

r, n = dict(leaves({k: v for k, v in ref.items() if k != "meta"})), dict(leaves({k: v for k, v in new.items() if k != "meta"}))
rows = []
for p in sorted(set(r) | set(n)):
    a, b = r.get(p, "<missing>"), n.get(p, "<missing>")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-6
        diff = b - a
    else:
        same, diff = a == b, None
    rows.append({"value": p, "reference": a, "recomputed": b, "diff": diff, "status": "MATCH" if same else "MISMATCH"})
check = pd.DataFrame(rows)
print("Reference versions :", ref["meta"]["versions"])
print("Recomputed versions:", new["meta"]["versions"])
print(check.status.value_counts().to_string())"""),
    code("""pd.set_option("display.max_rows", 50)
check[check.status == "MISMATCH"]"""),
    md("## 5. Main tables (recomputed)"),
    code("""print("Table 3"); display(pd.DataFrame(new["nested_cv"]).T[["macro_f1_mean", "macro_f1_sd", "bal_acc_mean", "roc_auc", "pr_auc", "brier", "macro_f1_tuned_mean"]].round(3))
print("Table 4"); display(pd.DataFrame(new["bootstrap"]).T)
print("Table 5"); display(pd.DataFrame(new["lcgo"]).round(3))
print("Table 6"); display(pd.DataFrame({k: {m: f"{f:.3f} / {a:.3f}" for m, (f, a) in v.items()} for k, v in new["sensitivity"]["table"].items()}))
print("Table 7"); display(pd.DataFrame(new["shap"]["table"]).round(3))
print("Table 8"); display(pd.DataFrame(new["routing"]["table"]).round(3))
print("Table 9"); display(pd.DataFrame(new["case_study"]).T)"""),
    code("""check.to_csv("results/comparison_report.csv", index=False)
print("Saved results/comparison_report.csv")"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, "notebooks/reproduce_results.ipynb")
print("written notebooks/reproduce_results.ipynb")
