#!/usr/bin/env python3
"""Draws the three figures of test14 from results/tables. Usage: make_figures.py <tables folder> <figures folder>"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

TABLES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160})


def rows(name):
    return list(csv.DictReader(open(TABLES / name)))


def lookup(data, **keys):
    for row in data:
        if all(row[k] == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def from_base():
    internal, external = rows('jf_paired.csv'), rows('external_paired.csv')
    labels = (('legacy', 'legacy (MSE)'), ('mlp_context', 'mlp_context (MSE)'), ('e2e_ctx_f', 'e2e_ctx (fine-tuned)'), ('e2e_ctx_pin7', 'e2e_ctx + pinned record'))
    panels = [('MOSE dev', lambda m: lookup(internal, group='MOSE dev', videos_scope='all', method=m, reference='target_ref')),
              ('LVOS val', lambda m: lookup(internal, group='LVOS val', videos_scope='all', method=m, reference='target_ref')),
              ('M3-VOS', lambda m: lookup(external, dataset='m3vos', switch='all switches', method=m, reference='base')),
              ('PUMaVOS', lambda m: lookup(external, dataset='pumavos', switch='all switches', method=m, reference='base'))]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5), sharey=True, sharex=True)
    for ax, (title, get) in zip(axes, panels):
        for y, (method, _) in enumerate(reversed(labels)):
            row = get(method)
            d, low, high = float(row['diff']), float(row['ci_low']), float(row['ci_high'])
            ax.plot([low, high], [y, y], color=BLUE, lw=2, solid_capstyle='round')
            ax.plot(d, y, 'o', ms=8, color=BLUE, mec='white', mew=2)
            ax.text(d, y + .2, f'{d:+.2f}', ha='center', va='bottom', fontsize=9, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis='x', color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylim(-.6, 3.8)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(4), [x[1] for x in reversed(labels)])
    fig.supxlabel('J&F difference from the Base+ state, points (95% CI over videos)', fontsize=11, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / 'jf_vs_base.png', facecolor='white')


def by_switch():
    data = rows('paper_pumavos_levels.csv')
    series = (('Full Replay', 'Full Replay', AQUA), ('e2e_ctx_pin7', 'e2e_ctx + pinned record', YELLOW), ('e2e_ctx_f', 'e2e_ctx', ORANGE), ('paper Translator', 'paper Translator', BLUE))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = (25, 50, 75)
    for method, label, color in series:
        row = lookup(data, method=method)
        y = [float(row[f'switch_{n}']) for n in x]
        ax.plot(x, y, '-o', lw=2, ms=8, color=color, mec='white', mew=2, label=label)
        ax.annotate(label, (75, y[-1]), xytext=(10, 0), textcoords='offset points', va='center', fontsize=9.5, color=INK)
    ax.set_xticks(x, ['25%', '50% (paper)', '75%'])
    ax.set_xlim(22, 92)
    ax.set_ylabel('J&F')
    ax.set_xlabel('Switch point')
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc='upper center', bbox_to_anchor=(.5, 1.17), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / 'pumavos_by_switch.png', facecolor='white')


def against_paper():
    data = rows('paper_pumavos_paired.csv')
    labels = (('e2e_ctx_f', 'e2e_ctx'), ('e2e3_ctx_f', 'e2e3_ctx'), ('e2e_ctx_pin7', 'e2e_ctx + pin7'), ('e2e3_ctx_pin5', 'e2e3_ctx + pin5'))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for scope, label, color, shift in (('50', 'switch at 50%', BLUE, .17), ('all switches', 'mean of 25, 50, 75%', ORANGE, -.17)):
        for y, (method, _) in enumerate(reversed(labels)):
            row = lookup(data, switch=scope, method=method, paper_method='paper Translator')
            d, low, high = float(row['diff']), float(row['ci_low']), float(row['ci_high'])
            ax.plot([low, high], [y + shift] * 2, color=color, lw=2, solid_capstyle='round')
            ax.plot(d, y + shift, 'o', ms=8, color=color, mec='white', mew=2, label=label if y == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(4), [x[1] for x in reversed(labels)])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('J&F difference from the paper Translator, PUMaVOS, points (95% CI over videos)')
    ax.legend(frameon=False, ncols=2, loc='upper center', bbox_to_anchor=(.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT / 'pumavos_vs_paper.png', facecolor='white')


from_base()
by_switch()
against_paper()
