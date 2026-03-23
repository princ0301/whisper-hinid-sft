import re
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter
from itertools import zip_longest
 
df = pd.read_csv("/teamspace/studios/this_studio/Question 4 - Task.csv")
 
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')

MODEL_COLS = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']
MODELS = MODEL_COLS

print("LATTICE-BASED WER — HINDI ASR EVALUATION")
 
print(f"Segments: {len(df)}")
print(f"Models:   {MODELS}")
print()
 
def normalize(text):
    """
    Normalize Hindi text for fair comparison:
    - NFC unicode normalization (matra consistency)
    - Remove punctuation (।,!?.,;:-–—)
    - Collapse whitespace
    - Lowercase (handles Roman script English)
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = unicodedata.normalize('NFC', text)
    # Remove punctuation but keep Devanagari, Latin, digits, spaces
    text = re.sub(r'[^\u0900-\u097F\s a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text.split() if text else []

def levenshtein(ref, hyp):
    """Return (edits, substitutions, deletions, insertions)"""
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                    dp[i][j-1],    # insertion
                                    dp[i-1][j-1])  # substitution
    return dp[n][m]


def wer(ref_words, hyp_words):
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)

def align_sequences(seqs):
    if not seqs:
        return []
    max_len = max(len(s) for s in seqs)
    padded  = [s + [None]*(max_len - len(s)) for s in seqs]
    
    bins = []
    for col_idx in range(max_len):
        col_words = [row[col_idx] for row in padded if row[col_idx] is not None]
        if col_words:
            bins.append(set(col_words))
    return bins


def build_lattice(human_ref, model_outputs):
    ref_words = human_ref
    all_seqs = [human_ref] + model_outputs
    n_models = len(model_outputs)
     
    lattice = []
     
    alignments = []
    for hyp in model_outputs:
        aligned = align_to_ref(ref_words, hyp)
        alignments.append(aligned)
     
    n_positions = len(ref_words)
    
    for pos in range(n_positions):
        ref_word = ref_words[pos]
         
        model_words_here = []
        for aligned in alignments:
            if pos < len(aligned):
                w = aligned[pos]
                if w is not None:
                    model_words_here.append(w)
         
        counter      = Counter(model_words_here)
        total_models = len(model_outputs)
        
        bin_words = set()
        bin_words.add(ref_word)  

        for w in model_words_here:
            if w:
                bin_words.add(w)
         
        majority_word   = counter.most_common(1)[0][0] if counter else ref_word
        majority_count  = counter.most_common(1)[0][1] if counter else 0
        majority_agrees = majority_count >= (total_models // 2 + 1)  # > 50%
        
        reference_is_wrong = (
            majority_agrees and
            majority_word != ref_word and
            majority_count >= 4  
        )
        
        lattice.append({
            'position': pos,
            'ref_word': ref_word,
            'alternatives': bin_words,
            'majority_word': majority_word,
            'majority_count': majority_count,
            'reference_wrong': reference_is_wrong,
            'optional': False 
        })
     
    for pos in range(n_positions - 1):
        inserted = []
        for aligned in alignments:
            pass  # 
    
    return lattice


def align_to_ref(ref, hyp):
    if not ref:
        return []
    if not hyp:
        return [None] * len(ref)
    
    n, m = len(ref), len(hyp)
    # DP matrix
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = -i
    for j in range(m+1): dp[0][j] = -j
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + (1 if ref[i-1] == hyp[j-1] else -1)
            delete = dp[i-1][j] - 1
            insert = dp[i][j-1] - 1
            dp[i][j] = max(match, delete, insert)
     
    aligned_hyp = []
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (1 if ref[i-1] == hyp[j-1] else -1):
            path.append(('match', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] - 1:
            path.append(('delete', ref[i-1], None))
            i -= 1
        else:
            path.append(('insert', None, hyp[j-1]))
            j -= 1
    path.reverse()
     
    result = []
    for op, r, h in path:
        if op in ('match',):
            result.append(h)
        elif op == 'delete':
            result.append(None)
    
    return result

# Lattice-based WER 
def lattice_wer(lattice, hyp_words):
    if not lattice:
        return 0.0
    
    # Align hypothesis to lattice positions
    ref_words   = [node['ref_word'] for node in lattice]
    aligned_hyp = align_to_ref(ref_words, hyp_words)
    
    errors = 0
    total  = len(lattice)
    
    for i, node in enumerate(lattice):
        hyp_word = aligned_hyp[i] if i < len(aligned_hyp) else None
        
        if hyp_word is None:
            # Deletion
            if not node['optional']:
                errors += 1
        elif hyp_word in node['alternatives']:
            # Match against any alternative → NO ERROR
            pass
        else:
            # True substitution error
            errors += 1
     
    n_insertions = max(0, len(hyp_words) - len(lattice))
    errors += n_insertions
    
    return errors / total if total > 0 else 0.0
 
results = []

for idx, row in df.iterrows():
    ref_raw  = row['Human']
    ref_words = normalize(ref_raw)
    
    model_outputs = {}
    model_words   = {}
    for col in MODEL_COLS:
        raw = row.get(col, '')
        model_outputs[col] = raw
        model_words[col]   = normalize(raw)
     
    lattice = build_lattice(
        ref_words,
        [model_words[c] for c in MODEL_COLS]
    )
     
    n_overrides = sum(1 for node in lattice if node['reference_wrong'])
    
    seg_result = {
        'segment': idx,
        'ref': ref_raw,
        'ref_len': len(ref_words),
        'n_overrides': n_overrides,
        'lattice_size': len(lattice),
    }
    
    for col in MODEL_COLS:
        hyp   = model_words[col]
        std_w = wer(ref_words, hyp)
        lat_w = lattice_wer(lattice, hyp)
        seg_result[f'{col}_std_wer']     = round(std_w, 4)
        seg_result[f'{col}_lattice_wer'] = round(lat_w, 4)
        seg_result[f'{col}_improved']    = lat_w < std_w
    
    results.append(seg_result)

results_df = pd.DataFrame(results)
 
print("RESULTS: STANDARD WER vs LATTICE WER (per model)")
print(f"{'Model':<12} {'Std WER':>10} {'Lattice WER':>12} {'Reduction':>10} {'Segments Improved':>18}")

summary = []
for col in MODEL_COLS:
    std_wers = results_df[f'{col}_std_wer'].tolist()
    lat_wers = results_df[f'{col}_lattice_wer'].tolist()
    mean_std = np.mean(std_wers)
    mean_lat = np.mean(lat_wers)
    reduction = mean_std - mean_lat
    n_improved = results_df[f'{col}_improved'].sum()
    
    print(f"{col:<12} {mean_std*100:>9.2f}% {mean_lat*100:>11.2f}% {reduction*100:>+9.2f}%  {n_improved:>10}/{len(results_df)}")
    summary.append({
        'model': col,
        'std_wer': round(mean_std, 4),
        'lattice_wer': round(mean_lat, 4),
        'reduction': round(reduction, 4),
        'n_improved': int(n_improved)
    })
 
 
total_overrides = results_df['n_overrides'].sum()
total_positions = results_df['lattice_size'].sum()
print(f"\nMajority overrides (reference corrected by models): {total_overrides}/{total_positions} positions ({total_overrides/total_positions*100:.1f}%)")
 
print("\n" + "="*70)
print("EXAMPLE SEGMENTS WHERE LATTICE REDUCED WER")
print("="*70)

improved_segs = results_df[
    results_df[[f'{c}_improved' for c in MODEL_COLS]].any(axis=1)
].head(5)

for _, row in improved_segs.iterrows():
    idx = int(row['segment'])
    print(f"\nSegment {idx}: {df.iloc[idx]['Human'][:80]}")
    print(f"  Lattice bins with alternatives: {row['n_overrides']} majority overrides")
    for col in MODEL_COLS:
        std = row[f'{col}_std_wer']
        lat = row[f'{col}_lattice_wer']
        tag = "↓ improved" if lat < std else "  same"
        print(f"  {col:<10}: std={std*100:.1f}%  lattice={lat*100:.1f}%  {tag}")
 
ref_words = normalize(df.iloc[1]['Human'])
model_words_sample = [normalize(df.iloc[1][c]) for c in MODEL_COLS]
sample_lattice = build_lattice(ref_words, model_words_sample)

print(f"\nReference: {' '.join(ref_words)}")
print(f"\nLattice bins:")
for i, node in enumerate(sample_lattice):
    alts = node['alternatives'] - {node['ref_word']}
    override = "MAJORITY OVERRIDE" if node['reference_wrong'] else ""
    alts_str = f" | alts: {alts}" if alts else ""
    print(f"Bin {i:2d}: [{node['ref_word']}]{alts_str}{override}")
 
results_df.to_csv("q4_results.csv", index=False)
pd.DataFrame(summary).to_csv("q4_summary.csv", index=False)
print(f"\nResults saved to /hindi_asr/q4_results.csv")
 