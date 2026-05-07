import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

edge_root = Path('/home/pablo/M2_DS/Secondary-Model/src/Output/Analysis/Edge_NoCal')
bt_root   = Path('/home/pablo/M2_DS/Secondary-Model/src/Output')

M1=['Tirex','Chronos2','Fincast','Kronos']
M2=['rf','autogluon','tabpfn','tabicl','ctts']
DIRS=['UP','DOWN']
GRANS=['1d','12h','8h','6h','4h','2h','1h','30m']

records = []
for m1 in M1:
    for m2 in M2:
        for d in DIRS:
            for g in GRANS:
                ep = edge_root/m1/m2/d/f'edge_summary_{g}.json'
                bp = bt_root/m1/m2/d/'Utility_Score_NoCal'/f'{g}_tp'/'analysis_summary.json'
                try:
                    entry = json.load(open(ep)).get(g,{})
                    bt    = json.load(open(bp))
                    tkey  = f'{m2}_temporal_all_features'
                    bkey  = f'{m2}_backtest_all_features'
                    val_sel  = bt[tkey]['Val_selective']
                    val_ret  = val_sel['mean_ret']
                    test_ret = bt[bkey]['m2_total_return']
                    if val_ret is None or test_ret is None: continue
                    if not val_sel.get('constraint_satisfied', False): continue
                    p   = np.array(entry.get('path_total_rets',[]), dtype=float)
                    srs = np.array(entry.get('path_sharpes',[]), dtype=float)
                    cv        = float(np.std(p)/(abs(np.mean(p))+1e-6)) if len(p)>1 else 99
                    mean_sr   = float(np.mean(srs)) if len(srs)>0 else -99
                    path_mean = float(np.mean(p)) if len(p)>0 else -99
                    b = bt[bkey]
                    records.append({
                        'val_pos':       int(val_ret  > 0),
                        'test_pos':      int(test_ret > 0),
                        'fp':            entry.get('frac_profitable', 0),
                        'med_sr':        entry.get('median_path_sharpe', -99),
                        'mean_sr':       mean_sr,
                        'pp_mean':       entry.get('path_sel_prec_mean', 0),
                        'pp_std':        entry.get('path_sel_prec_std', 99),
                        'cv':            cv,
                        'path_mean':     path_mean,
                        'val_mean_ret':  val_ret,
                        'val_tstat':     val_sel.get('t_stat', 0),
                        'val_constr':    int(bool(val_sel.get('constraint_satisfied', False))),
                        # portfolio bridge metrics
                        'test_ret':      test_ret,
                        'test_sharpe':   b.get('m2_sharpe', None),
                        'val_sharpe':    float(np.mean(srs)) if len(srs) > 0 else None,  # mean path SR
                        'med_path_sr':   entry.get('median_path_sharpe', None),
                        'm1':            m1, 'm2': m2, 'dir': d, 'gran': g,
                    })
                except: pass

N = len(records)
print(f'N={N}')

# ┏━━━━━━━━━━ Portfolio bridge correlation analysis ━━━━━━━━━━┓
def _corr(xs, ys):
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 5: return None, int(mask.sum())
    r = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
    return round(r, 4), int(mask.sum())

cpcv_predictors = {
    'frac_profitable':    [r['fp']          for r in records],
    'mean_path_SR':       [r['mean_sr']     for r in records],
    'median_path_SR':     [r['med_sr']      for r in records],
    'CV':                 [r['cv']          for r in records],
    'mean_path_ret':      [r['path_mean']   for r in records],
    'val_sel_mean_ret':   [r['val_mean_ret']for r in records],
    'val_sel_tstat':      [r['val_tstat']   for r in records],
}

targets = {
    'test_total_return':  [r['test_ret']    for r in records],
    'test_sharpe':        [r['test_sharpe'] if r['test_sharpe'] is not None else float('nan') for r in records],
    'test_ret_sign':      [r['test_pos']    for r in records],
}

corr_results = {}
print("\n[Bridge correlations: CPCV predictor → test portfolio metric]")
print(f"  {'Predictor':<22} {'→ test_return':>14} {'→ test_sharpe':>14} {'→ test_ret_sign':>16}")
print("  " + "-"*68)
for pred_name, pred_vals in cpcv_predictors.items():
    row = {}
    line = f"  {pred_name:<22}"
    for tgt_name, tgt_vals in targets.items():
        r_val, n = _corr(pred_vals, tgt_vals)
        row[tgt_name] = {'r': r_val, 'n': n}
        line += f"  {r_val:+.3f}(n={n})" if r_val is not None else "       N/A"
    corr_results[pred_name] = row
    print(line)

# Also test: does rank of val path SR predict rank of test sharpe? (Spearman)
from scipy.stats import spearmanr
print("\n[Spearman rank correlations]")
spearman_results = {}
for pred_name, pred_vals in cpcv_predictors.items():
    row = {}
    line = f"  {pred_name:<22}"
    for tgt_name, tgt_vals in targets.items():
        pv = np.array(pred_vals, dtype=float)
        tv = np.array(tgt_vals, dtype=float)
        mask = np.isfinite(pv) & np.isfinite(tv)
        if mask.sum() < 5:
            row[tgt_name] = None
            line += "       N/A"
            continue
        rho, pval = spearmanr(pv[mask], tv[mask])
        row[tgt_name] = {'rho': round(float(rho), 4), 'pval': round(float(pval), 4), 'n': int(mask.sum())}
        line += f"  {rho:+.3f}(p={pval:.3f})"
    spearman_results[pred_name] = row
    print(line)

out_json = edge_root / 'cpcv_bridge_correlations.json'
json.dump({'pearson': corr_results, 'spearman': spearman_results, 'N': N}, open(out_json, 'w'), indent=2)
print(f"\nSaved correlations -> {out_json}")

base_conditions = [
    ('fp≥0.6',      lambda r: r['fp'] >= 0.6),
    ('fp≥0.8',      lambda r: r['fp'] >= 0.8),
    ('meanSR>0.5',  lambda r: r['mean_sr'] > 0.5),
    ('meanSR>1.0',  lambda r: r['mean_sr'] > 1.0),
    ('meanSR≥1.5',  lambda r: r['mean_sr'] >= 1.5),
    ('medSR>0.5',   lambda r: r['med_sr'] > 0.5),
    ('medSR>1.0',   lambda r: r['med_sr'] > 1.0),
    ('medSR≥1.5',   lambda r: r['med_sr'] >= 1.5),
    ('CV<1.0',      lambda r: r['cv'] < 1.0),
    ('CV<0.5',      lambda r: r['cv'] < 0.5),
    ('prec≥0.52',   lambda r: r['pp_mean'] >= 0.52),
    ('pathMean>0',  lambda r: r['path_mean'] > 0),
    ('valRet>0',    lambda r: r['val_mean_ret'] > 0),
    ('tStat>1.5',   lambda r: r['val_tstat'] > 1.5),
    ('tStat>2',     lambda r: r['val_tstat'] > 2),
    ('tStat>3',     lambda r: r['val_tstat'] > 3),
    ('constr=True', lambda r: r['val_constr'] == 1),
]

def combine(fns):
    def f(r):
        return all(fn(r) for fn in fns)
    return f

# Groups where using two from the same group is redundant (stricter subsumes looser)
redundant_groups = [
    {'fp≥0.6', 'fp≥0.8'},
    {'meanSR>0.5', 'meanSR>1.0', 'meanSR≥1.5'},
    {'medSR>0.5', 'medSR>1.0', 'medSR≥1.5'},
    {'tStat>1.5', 'tStat>2', 'tStat>3'},
    # constr=True implies val mean_ret>0 and t>=t_min — combining them is redundant
    {'CV<1.0', 'CV<0.5'},
    {'constr=True', 'valRet>0'},
    {'constr=True', 'tStat>1.5'},
    {'constr=True', 'tStat>2'},
]

def has_redundancy(names):
    for group in redundant_groups:
        if len(group & set(names)) > 1:
            return True
    return False

all_filters = [('Baseline\n(no filter)', lambda r: True)]
for name, fn in base_conditions:
    all_filters.append((name, fn))
for (n1,f1),(n2,f2) in combinations(base_conditions, 2):
    if not has_redundancy([n1, n2]):
        all_filters.append((f'{n1} &\n{n2}', combine([f1,f2])))
for (n1,f1),(n2,f2),(n3,f3) in combinations(base_conditions, 3):
    if not has_redundancy([n1, n2, n3]):
        all_filters.append((f'{n1} &\n{n2} & {n3}', combine([f1,f2,f3])))

# Compute stats for all filters
def compute_stats(filters, out_key):
    TPs, FPs, FNs, TNs = [], [], [], []
    for _, fn in filters:
        sel = [r for r in records if fn(r)]
        rej = [r for r in records if not fn(r)]
        TPs.append(sum(r[out_key]==1 for r in sel))
        FPs.append(sum(r[out_key]==0 for r in sel))
        FNs.append(sum(r[out_key]==1 for r in rej))
        TNs.append(sum(r[out_key]==0 for r in rej))
    return TPs, FPs, FNs, TNs

TPs_t, FPs_t, FNs_t, TNs_t = compute_stats(all_filters, 'test_pos')

# Rank by test precision, keep top 30 + baseline always first
precisions_all = [TP/(TP+FP) if (TP+FP)>0 else -1 for TP,FP in zip(TPs_t, FPs_t)]
# baseline is index 0, rank rest by test precision descending
ranked_idx = [0] + sorted(range(1, len(all_filters)), key=lambda i: -precisions_all[i])
TOP_N = 30
keep_idx = ranked_idx[:TOP_N+1]  # baseline + top 30

filters = [all_filters[i] for i in keep_idx]

# Always append CV<0.5 & fp>=0.8 as the last filter
def _cv05_fp08(r): return r['cv'] < 0.5 and r['fp'] >= 0.8
filters.append(('CV<0.5 &\nfp≥0.8', _cv05_fp08))

splits = [
    ('VAL',  'val_pos',  'Val_selective mean_ret > 0'),
    ('TEST', 'test_pos', 'm2_total_return > 0'),
]

fig, axes = plt.subplots(2, 1, figsize=(26, 16), dpi=160)
fig.patch.set_facecolor('white')

for ax, (split, out_key, split_label) in zip(axes, splits):
    n_pos = sum(r[out_key] for r in records)
    n_neg = N - n_pos
    bar_w = 0.18
    TPs, FPs, FNs, TNs = compute_stats(filters, out_key)
    x = np.arange(len(filters))

    TPs_r = np.array(TPs)/N; FPs_r = np.array(FPs)/N
    FNs_r = np.array(FNs)/N; TNs_r = np.array(TNs)/N

    b1 = ax.bar(x-1.5*bar_w, TPs_r, bar_w, label='TP: selected & profitable',         color='#2ca02c', edgecolor='white')
    b2 = ax.bar(x-0.5*bar_w, FPs_r, bar_w, label='FP: selected & NOT profitable',      color='#d62728', edgecolor='white')
    b3 = ax.bar(x+0.5*bar_w, TNs_r, bar_w, label='TN: rejected & NOT profitable',      color='#1f77b4', edgecolor='white')
    b4 = ax.bar(x+1.5*bar_w, FNs_r, bar_w, label='FN: rejected & profitable (missed)', color='#ff7f0e', edgecolor='white')

    for bars, vals in [(b1,TPs),(b2,FPs),(b3,TNs),(b4,FNs)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                        str(v), ha='center', va='bottom', fontsize=7,
                        fontweight='bold', color='#111111')

    ax2 = ax.twinx()
    precisions = [TP/(TP+FP) if (TP+FP)>0 else np.nan for TP,FP in zip(TPs,FPs)]
    recalls    = [TP/(TP+FN) if (TP+FN)>0 else np.nan for TP,FN in zip(TPs,FNs)]
    accuracies = [(TP+TN)/N if N>0 else np.nan for TP,TN in zip(TPs,TNs)]
    ax2.plot(x, precisions, 'D--', color='#9467bd', lw=1.8, ms=6,
             label='Precision TP/(TP+FP)', zorder=5)
    ax2.plot(x, recalls,    's--', color='#8c564b', lw=1.8, ms=6,
             label='Recall TP/(TP+FN)', zorder=5)
    ax2.plot(x, accuracies, '^--', color='#17becf', lw=1.8, ms=6,
             label='Accuracy (TP+TN)/N', zorder=5)
    for xi,(p,r_,a) in enumerate(zip(precisions,recalls,accuracies)):
        if np.isfinite(p):
            ax2.text(xi-0.22, p+0.02, f'{p:.0%}', fontsize=6.5,
                     color='#9467bd', ha='center', fontweight='bold')
        if np.isfinite(r_):
            ax2.text(xi+0.0,  r_-0.05, f'{r_:.0%}', fontsize=6.5,
                     color='#8c564b', ha='center', fontweight='bold')
        if np.isfinite(a):
            ax2.text(xi+0.22, a+0.02, f'{a:.0%}', fontsize=6.5,
                     color='#17becf', ha='center', fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Precision / Recall / Accuracy', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax2.spines['top'].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in filters], fontsize=8, rotation=35, ha='right')
    ax.set_ylabel(f'Fraction of all configs (N={N})', fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#fafafa')
    ax.grid(axis='y', color='#dddddd', lw=0.6, zorder=0)

    # highlight baseline bar group with subtle background
    ax.axvspan(-0.5, 0.5, color='#ffffcc', alpha=0.5, zorder=0)

    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=8, framealpha=0.9, ncol=3)

    ax.text(0.01, 0.97,
            f'Actually profitable: {n_pos}/{N} ({n_pos/N:.1%})  |  '
            f'Actually not profitable: {n_neg}/{N} ({n_neg/N:.1%})',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.9))

    ax.set_title(f'► {split} split  —  Profitable = {split_label}',
                 fontsize=11, fontweight='bold', pad=8)

fig.suptitle(
    f'CPCV Filter Confusion Analysis  |  Top {TOP_N} filters by TEST precision (out of {len(all_filters)})  |  '
    f'N={N} configs  |  Ranked left→right by test precision',
    fontsize=11, fontweight='bold', y=1.01)

plt.tight_layout(h_pad=4.0)
out = edge_root / 'cpcv_edge_heatmap_test.png'
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved -> {out}')

# ┏━━━━━━━━━━ Tab vs CTTS comparison ━━━━━━━━━━┓
def _load_cell(m1, m2, d, g):
    bp = bt_root/m1/m2/d/'Utility_Score_NoCal'/f'{g}_tp'/'analysis_summary.json'
    ep = edge_root/m1/m2/d/f'edge_summary_{g}.json'
    try:
        bt_    = json.load(open(bp))
        entry_ = json.load(open(ep)).get(g, {})
        b_ = bt_[f'{m2}_backtest_all_features']
        t_ = bt_[f'{m2}_temporal_all_features']
        val_sel_ = t_['Val_selective']
        constr_  = bool(val_sel_.get('constraint_satisfied', False))
        p_       = np.array(entry_.get('path_total_rets', []), dtype=float)
        cv_      = float(np.std(p_) / (abs(np.mean(p_)) + 1e-6)) if len(p_) > 1 else 99.0
        return {
            'prec_delta': b_['m2_win_rate'] - b_['m1_win_rate'],
            'm2_return':  b_['m2_total_return'],
            'green':      constr_ and cv_ < 1.0,
        }
    except:
        return None

TAB_MODELS = ['rf', 'autogluon', 'tabpfn', 'tabicl']

t1_tab_wins = 0; t1_ctts_wins = 0; t1_tie = 0; t1_total = 0
t2_scA_tab = 0;  t2_scA_ctts = 0; t2_scA_tie = 0
t2_scB = 0; t2_scC = 0; t2_neither = 0

for m1 in M1:
    for d in DIRS:
        for g in GRANS:
            ctts = _load_cell(m1, 'ctts', d, g)
            tabs = {m2: _load_cell(m1, m2, d, g) for m2 in TAB_MODELS}

            # Table 1: best tab Δ Precision vs CTTS Δ Precision
            tab_deltas = [v['prec_delta'] for v in tabs.values() if v is not None]
            ctts_delta = ctts['prec_delta'] if ctts else None
            if tab_deltas and ctts_delta is not None:
                best_tab = max(tab_deltas)
                t1_total += 1
                if   best_tab > ctts_delta: t1_tab_wins  += 1
                elif best_tab < ctts_delta: t1_ctts_wins += 1
                else:                       t1_tie       += 1

            # Table 2: reliability-aware M2 return
            green_tabs = {m2: v for m2, v in tabs.items() if v is not None and v['green']}
            ctts_green = ctts is not None and ctts['green']

            if ctts_green and green_tabs:
                best_tab_ret = max(v['m2_return'] for v in green_tabs.values())
                ctts_ret     = ctts['m2_return']
                if   best_tab_ret > ctts_ret: t2_scA_tab  += 1
                elif best_tab_ret < ctts_ret: t2_scA_ctts += 1
                else:                         t2_scA_tie  += 1
            elif not ctts_green and green_tabs: t2_scB    += 1
            elif ctts_green and not green_tabs: t2_scC    += 1
            else:                               t2_neither += 1

print('\n=== TABLE 1: best tab Δ Precision vs CTTS Δ Precision ===')
print(f'  Total cells : {t1_total}')
print(f'  Tab wins    : {t1_tab_wins}  ({t1_tab_wins/t1_total:.1%})')
print(f'  CTTS wins   : {t1_ctts_wins} ({t1_ctts_wins/t1_total:.1%})')
print(f'  Tie         : {t1_tie}')

print('\n=== TABLE 2: reliability-aware portfolio comparison ===')
print(f'  Total cells                          : {len(M1)*len(DIRS)*len(GRANS)}')
print(f'  Scenario A (both green) -> tab wins  : {t2_scA_tab}')
print(f'  Scenario A (both green) -> ctts wins : {t2_scA_ctts}')
print(f'  Scenario A (both green) -> tie       : {t2_scA_tie}')
print(f'  Scenario B (ctts RED, tab GREEN)     : {t2_scB}  (tab wins by default)')
print(f'  CTTS green, no green tab             : {t2_scC}  (ctts wins by default)')
print(f'  Neither green                        : {t2_neither}')
print(f'\n  Tab  wins (A+B) : {t2_scA_tab + t2_scB}')
print(f'  CTTS wins (A+C) : {t2_scA_ctts + t2_scC}')

tab_ctts_results = {
    'table1': {
        'total': t1_total, 'tab_wins': t1_tab_wins, 'ctts_wins': t1_ctts_wins, 'tie': t1_tie,
        'tab_win_pct': round(t1_tab_wins/t1_total, 4), 'ctts_win_pct': round(t1_ctts_wins/t1_total, 4),
    },
    'table2': {
        'total': len(M1)*len(DIRS)*len(GRANS),
        'scA_tab_wins': t2_scA_tab, 'scA_ctts_wins': t2_scA_ctts, 'scA_tie': t2_scA_tie,
        'scB_ctts_red_tab_green': t2_scB,
        'scC_ctts_green_no_tab': t2_scC,
        'neither_green': t2_neither,
        'total_tab_wins': t2_scA_tab + t2_scB,
        'total_ctts_wins': t2_scA_ctts + t2_scC,
    }
}
out_json2 = edge_root / 'tab_vs_ctts_comparison.json'
json.dump(tab_ctts_results, open(out_json2, 'w'), indent=2)
print(f'\nSaved -> {out_json2}')
