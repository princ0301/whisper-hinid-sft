import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "q4"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH   = DATA_DIR / "Question 4 - Task.csv"
MODEL_COLS = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\u0900-\u097F\s a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text.split() if text else []


def levenshtein(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def standard_wer(ref, hyp):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)


def align_to_ref(ref, hyp):
    if not ref: return []
    if not hyp: return [None] * len(ref)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = -i
    for j in range(m + 1): dp[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match  = dp[i - 1][j - 1] + (1 if ref[i - 1] == hyp[j - 1] else -1)
            dp[i][j] = max(match, dp[i - 1][j] - 1, dp[i][j - 1] - 1)
    i, j, path = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (1 if ref[i - 1] == hyp[j - 1] else -1):
            path.append(('match', hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] - 1:
            path.append(('del', None)); i -= 1
        else:
            path.append(('ins', hyp[j - 1])); j -= 1
    path.reverse()
    return [h for op, h in path if op in ('match', 'del')]


def build_lattice(ref, model_outputs):
    alignments = [align_to_ref(ref, m) for m in model_outputs]
    lattice    = []
    for pos in range(len(ref)):
        model_words = [a[pos] for a in alignments if pos < len(a) and a[pos] is not None]
        counter = Counter(model_words)
        top_word, top_count = counter.most_common(1)[0] if counter else (ref[pos], 0)
        alts = set([ref[pos]] + [w for w in model_words if w])
        ref_wrong = top_count >= 4 and top_word != ref[pos]
        lattice.append({'ref': ref[pos], 'alts': alts, 'ref_wrong': ref_wrong, 'majority': top_word, 'majority_count': top_count})
    return lattice


def lattice_wer(lattice, hyp):
    if not lattice: return 0.0
    ref = [n['ref'] for n in lattice]
    aligned = align_to_ref(ref, hyp)
    errors = 0
    for i, node in enumerate(lattice):
        hw = aligned[i] if i < len(aligned) else None
        if hw is None:
            errors += 1
        elif hw not in node['alts']:
            errors += 1
    errors += max(0, len(hyp) - len(lattice))
    return errors / len(lattice)


df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
print(f"Segments: {len(df)} | Models: {MODEL_COLS}")

results = []
for idx, row in df.iterrows():
    ref = normalize(row['Human'])
    mw = {c: normalize(row.get(c, '')) for c in MODEL_COLS}
    lat = build_lattice(ref, [mw[c] for c in MODEL_COLS])
    r = {'segment': idx, 'reference': row['Human'], 'n_overrides': sum(1 for n in lat if n['ref_wrong'])}
    for c in MODEL_COLS:
        r[f'{c}_std_wer'] = round(standard_wer(ref, mw[c]), 4)
        r[f'{c}_lattice_wer'] = round(lattice_wer(lat, mw[c]), 4)
        r[f'{c}_improved'] = r[f'{c}_lattice_wer'] < r[f'{c}_std_wer']
    results.append(r)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "lattice_results.csv", index=False)

summary = []
print(f"\n{'='*65}")
print(f"{'Model':<12} {'Std WER':>10} {'Lattice WER':>12} {'Reduction':>10} {'Improved':>10}")
print(f"{'-'*65}")
for c in MODEL_COLS:
    s = results_df[f'{c}_std_wer'].mean()
    l = results_df[f'{c}_lattice_wer'].mean()
    imp = results_df[f'{c}_improved'].sum()
    print(f"{c:<12} {s*100:>9.2f}% {l*100:>11.2f}% {(s-l)*100:>+9.2f}%  {imp:>6}/{len(results_df)}")
    summary.append({'model': c, 'std_wer': round(s, 4), 'lattice_wer': round(l, 4), 'reduction': round(s - l, 4), 'n_improved': int(imp)})
print(f"{'='*65}")

total_overrides = results_df['n_overrides'].sum()
total_positions = sum(len(normalize(r['Human'])) for _, r in df.iterrows())
print(f"Majority overrides: {total_overrides}/{total_positions} ({total_overrides/total_positions*100:.1f}%)")

pd.DataFrame(summary).to_csv(OUT_DIR / "lattice_summary.csv", index=False)
print(f"Outputs saved to: {OUT_DIR}")