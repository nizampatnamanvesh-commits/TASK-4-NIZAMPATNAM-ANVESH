# Task 4: Data Visualization
# Decodelabs Data Science Internship
# Dataset: titanic_cleaned.csv (from Task 2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv('titanic_cleaned.csv')
df['age_group'] = pd.cut(df['age'],
                         bins=[0,12,18,35,60,100],
                         labels=['Child','Teen','Young Adult','Middle Age','Senior'])

print("=" * 60)
print("         TASK 4: DATA VISUALIZATION")
print("=" * 60)
print(f"\n  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("  Generating 8 professional charts...\n")

# ── Colour palette ────────────────────────────────────────────────────────────
SURVIVE   = '#27ae60'
NOT_SURV  = '#e74c3c'
BLUE      = '#2980b9'
GOLD      = '#f39c12'
PURPLE    = '#8e44ad'
BG        = '#f8f9fa'
DARK      = '#2c3e50'

# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Survival Overview (2×2)
# ════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Titanic Survival Analysis — Overview',
              fontsize=16, fontweight='bold', color=DARK, y=0.98)

# ── Chart 1: Survival Donut ───────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor(BG)
survived   = df['survived'].sum()
not_surv   = len(df) - survived
wedges, texts, autotexts = ax.pie(
    [survived, not_surv],
    labels=['Survived', 'Did Not Survive'],
    colors=[SURVIVE, NOT_SURV],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2}
)
for t in autotexts:
    t.set_fontsize(11); t.set_fontweight('bold'); t.set_color('white')
ax.set_title('Overall Survival Rate', fontweight='bold', color=DARK, pad=12)
ax.text(0, 0, f'{survived}\n/{len(df)}', ha='center', va='center',
        fontsize=13, fontweight='bold', color=DARK)

# ── Chart 2: Survival by Gender (grouped bar) ────────────────
ax = axes[0, 1]
ax.set_facecolor(BG)
sex_data = df.groupby('sex')['survived'].value_counts(normalize=True).mul(100).unstack()
x        = np.arange(len(sex_data))
width    = 0.35
b1 = ax.bar(x - width/2, sex_data[0], width, label='Not Survived',
            color=NOT_SURV, edgecolor='white', linewidth=1.2)
b2 = ax.bar(x + width/2, sex_data[1], width, label='Survived',
            color=SURVIVE,  edgecolor='white', linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(['Female', 'Male'], fontsize=11)
ax.set_ylabel('Percentage (%)', color=DARK)
ax.set_title('Survival Rate by Gender', fontweight='bold', color=DARK, pad=12)
ax.legend(); ax.set_ylim(0, 85)
ax.spines[['top','right']].set_visible(False)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1,
            f'{h:.0f}%', ha='center', va='bottom', fontsize=9, color=DARK)

# ── Chart 3: Survival by Passenger Class ─────────────────────
ax = axes[1, 0]
ax.set_facecolor(BG)
cls_surv = df.groupby('pclass')['survived'].mean() * 100
bar_cols  = [GOLD, BLUE, NOT_SURV]
bars = ax.bar(['1st Class','2nd Class','3rd Class'], cls_surv.values,
              color=bar_cols, edgecolor='white', linewidth=1.5, width=0.5)
ax.set_ylabel('Survival Rate (%)', color=DARK)
ax.set_title('Survival Rate by Passenger Class', fontweight='bold', color=DARK, pad=12)
ax.set_ylim(0, 70)
ax.spines[['top','right']].set_visible(False)
ax.axhline(df['survived'].mean()*100, color=DARK, linestyle='--',
           linewidth=1, alpha=0.5, label='Overall avg')
ax.legend(fontsize=9)
for bar, val in zip(bars, cls_surv.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold', color=DARK)

# ── Chart 4: Survival by Age Group ───────────────────────────
ax = axes[1, 1]
ax.set_facecolor(BG)
age_surv = df.groupby('age_group', observed=True)['survived'].mean() * 100
palette  = [BLUE, PURPLE, GOLD, '#16a085', NOT_SURV]
bars = ax.bar(age_surv.index, age_surv.values,
              color=palette[:len(age_surv)], edgecolor='white', linewidth=1.5, width=0.55)
ax.set_ylabel('Survival Rate (%)', color=DARK)
ax.set_title('Survival Rate by Age Group', fontweight='bold', color=DARK, pad=12)
ax.set_ylim(0, 70)
ax.tick_params(axis='x', labelrotation=15)
ax.spines[['top','right']].set_visible(False)
for bar, val in zip(bars, age_surv.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=DARK)

plt.tight_layout()
fig1.savefig('task4_fig1_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("  ✓ Figure 1 saved → task4_fig1_overview.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Distributions & Relationships (2×2)
# ════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor(BG)
fig2.suptitle('Titanic — Distributions & Relationships',
              fontsize=16, fontweight='bold', color=DARK, y=0.98)

# ── Chart 5: Age KDE by survival ─────────────────────────────
ax = axes[0, 0]
ax.set_facecolor(BG)
for label, color, val in [('Survived', SURVIVE, 1), ('Not Survived', NOT_SURV, 0)]:
    data = df[df['survived'] == val]['age']
    ax.hist(data, bins=30, alpha=0.55, color=color, edgecolor='white',
            label=label, density=True)
    # smooth line
    from numpy import convolve, ones
    counts, edges = np.histogram(data, bins=30, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth  = convolve(counts, ones(5)/5, mode='same')
    ax.plot(centers, smooth, color=color, linewidth=2)
ax.set_xlabel('Age', color=DARK); ax.set_ylabel('Density', color=DARK)
ax.set_title('Age Distribution by Survival', fontweight='bold', color=DARK, pad=12)
ax.legend(); ax.spines[['top','right']].set_visible(False)

# ── Chart 6: Fare boxplot by class ───────────────────────────
ax = axes[0, 1]
ax.set_facecolor(BG)
cls_data   = [df[df['pclass'] == c]['fare'].clip(upper=200).values for c in [1,2,3]]
bp = ax.boxplot(cls_data, tick_labels=['1st Class','2nd Class','3rd Class'],
                patch_artist=True, notch=False,
                medianprops={'color':'white','linewidth':2.5},
                whiskerprops={'color':DARK,'linewidth':1.2},
                capprops={'color':DARK,'linewidth':1.5},
                flierprops={'marker':'o','markerfacecolor':DARK,'markersize':4,'alpha':0.4})
for patch, color in zip(bp['boxes'], [GOLD, BLUE, NOT_SURV]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_ylabel('Fare (£, capped at 200)', color=DARK)
ax.set_title('Fare Distribution by Passenger Class', fontweight='bold', color=DARK, pad=12)
ax.spines[['top','right']].set_visible(False)

# ── Chart 7: Heatmap — Class × Gender survival ───────────────
ax = axes[1, 0]
ax.set_facecolor(BG)
pivot = df.pivot_table(values='survived', index='pclass', columns='sex', aggfunc='mean') * 100
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=80)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Female', 'Male'], fontsize=11)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['1st Class','2nd Class','3rd Class'], fontsize=10)
ax.set_title('Survival Rate Heatmap\n(Class × Gender)', fontweight='bold', color=DARK, pad=8)
plt.colorbar(im, ax=ax, label='Survival Rate (%)')
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f'{pivot.values[i,j]:.1f}%',
                ha='center', va='center', fontsize=13,
                fontweight='bold', color='white')

# ── Chart 8: Stacked bar — embarkation port ──────────────────
ax = axes[1, 1]
ax.set_facecolor(BG)
emb = df.groupby('embarked')['survived'].value_counts().unstack().fillna(0)
emb.index = ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)']
bottom = np.zeros(len(emb))
for val, color, label in [(0, NOT_SURV, 'Not Survived'), (1, SURVIVE, 'Survived')]:
    vals = emb[val].values
    bars = ax.bar(emb.index, vals, bottom=bottom, color=color,
                  edgecolor='white', linewidth=1.2, label=label, width=0.5)
    for bar, v, b in zip(bars, vals, bottom):
        if v > 10:
            ax.text(bar.get_x() + bar.get_width()/2, b + v/2,
                    f'{v:.0f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    bottom += vals
ax.set_ylabel('Number of Passengers', color=DARK)
ax.set_title('Passengers by Embarkation Port\n(Survival Stacked)', fontweight='bold', color=DARK, pad=8)
ax.legend(); ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='x', labelrotation=10)

plt.tight_layout()
fig2.savefig('task4_fig2_distributions.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("  ✓ Figure 2 saved → task4_fig2_distributions.png")

# ── Console summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 VISUALIZATION SUMMARY")
print("=" * 60)
summary = """
  Figure 1 — Survival Overview (4 charts):
    Chart 1 : Donut chart  — overall survival split
    Chart 2 : Grouped bar  — survival rate by gender
    Chart 3 : Bar chart    — survival rate by class
    Chart 4 : Bar chart    — survival rate by age group

  Figure 2 — Distributions & Relationships (4 charts):
    Chart 5 : Histogram    — age distribution by survival
    Chart 6 : Box plot     — fare spread per class
    Chart 7 : Heatmap      — survival by class × gender
    Chart 8 : Stacked bar  — embarkation port breakdown

  KEY VISUAL INSIGHTS:
    • Females survived at nearly 2× the rate of males
    • 1st class passengers had the strongest survival odds
    • Fare box plot reveals extreme outliers in 1st class
    • Southampton had the most passengers but mixed survival
    • Heatmap shows 1st class females had the highest rate
"""
print(summary)
print("=" * 60)
print("  Task 4 Complete ✓")
print("=" * 60)
