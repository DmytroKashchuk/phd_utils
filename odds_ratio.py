import pandas as pd
import numpy as np

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Load
path = "/Users/dmk6603/Documents/ransom/25.1-big_query_httparchive_collection/data/muni_tech_pivot_tech_2_m_after.csv"
df = pd.read_csv(path, low_memory=False)

# Clean hacked to 0/1 ints
df["hacked"] = pd.to_numeric(df["hacked"], errors="coerce").fillna(0).astype(int)
df = df[df["hacked"].isin([0, 1])].copy()

# Identify technology columns: everything except metadata
meta_cols = {"website", "hacked", "group"}
tech_cols = [c for c in df.columns if c not in meta_cols]

# Make sure tech cols are 0/1 ints (treat NaN as 0)
for c in tech_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

def odds_ratio_with_ci(a, b, c, d, correction=0.5):
    """
    2x2 table:
               hacked=1   hacked=0
    tech=1         a         b
    tech=0         c         d
    """
    # Haldane–Anscombe correction if any cell is 0 (or always, if you prefer)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + correction, b + correction, c + correction, d + correction

    or_val = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = np.exp(np.log(or_val) - 1.96 * se)
    hi = np.exp(np.log(or_val) + 1.96 * se)
    return or_val, lo, hi

results = []
hacked_mask = df["hacked"] == 1
not_hacked_mask = df["hacked"] == 0

for tech in tech_cols:
    tech_mask = df[tech] == 1
    no_tech_mask = df[tech] == 0

    a = int((tech_mask & hacked_mask).sum())
    b = int((tech_mask & not_hacked_mask).sum())
    c = int((no_tech_mask & hacked_mask).sum())
    d = int((no_tech_mask & not_hacked_mask).sum())

    # Fisher exact: table is [[a,b],[c,d]]
    # "two-sided" is safest; switch to "greater" if you only care about tech being more common in hacked
    p = fisher_exact([[a, b], [c, d]], alternative="two-sided")[1]

    or_val, lo, hi = odds_ratio_with_ci(a, b, c, d)

    results.append({
        "technology": tech,
        "a_tech_hacked": a,
        "b_tech_not_hacked": b,
        "c_no_tech_hacked": c,
        "d_no_tech_not_hacked": d,
        "odds_ratio": or_val,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_value": p,
        "tech_prevalence": float(tech_mask.mean()),
    })

res = pd.DataFrame(results)

# Multiple-testing correction (FDR)
res["p_fdr_bh"] = multipletests(res["p_value"], method="fdr_bh")[1]

# Sort: strongest associations first (or by significance)
res_sorted = res.sort_values(["odds_ratio"], ascending=False)

# Inspect top hits
print(f"Top 10 technologies associated with hacked municipalities 3 months before the attack: ")
print(res_sorted.head(10)[["technology", "odds_ratio", "ci95_low", "ci95_high", "p_value", "p_fdr_bh",
                           "a_tech_hacked", "b_tech_not_hacked", "tech_prevalence", "c_no_tech_hacked", "d_no_tech_not_hacked"]])

# Save
out_path = "/Users/dmk6603/Documents/ransom/25.1-big_query_httparchive_collection/data/muni_tech_odds_ratios.csv"
res_sorted.to_csv(out_path, index=False)
print("Saved:", out_path)
