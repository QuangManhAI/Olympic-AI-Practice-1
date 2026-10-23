# Build 5-fold StratifiedGroupKFold split: augmented copies (xxx_1.mp4...) stay with their original.
import os, re, pickle, unicodedata
import pandas as pd
import numpy as np

ROOT = 'dataset/train'
mapping = pickle.load(open('dataset/label_mapping.pkl', 'rb'))
mapping = {unicodedata.normalize('NFC', k): v for k, v in mapping.items()}

rows = []
for label in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, label)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith('.mp4'):
            rows.append({'path': os.path.join(d, f),
                         'label': unicodedata.normalize('NFC', label),
                         'label_idx': mapping[unicodedata.normalize('NFC', label)],
                         'group': re.sub(r'_\d+$', '', f[:-4]),
                         'is_orig': re.search(r'_\d+$', f[:-4]) is None})
df = pd.DataFrame(rows)

# Each group has one label, so stratify exactly: per class, shuffle groups and deal them round-robin.
# The starting fold rotates per class so fold sizes stay balanced.
rng = np.random.default_rng(42)
group_fold, offset = {}, 0
for _, g in df.groupby('label_idx'):
    groups = g.group.unique()
    rng.shuffle(groups)
    for i, grp in enumerate(groups):
        group_fold[grp] = (i + offset) % 5
    offset += len(groups)
df['fold'] = df.group.map(group_fold)
df.to_csv('work/folds.csv', index=False)

assert df.groupby('group').fold.nunique().max() == 1
orig = df[df.is_orig]
per = orig.groupby(['fold', 'label_idx']).size().unstack(fill_value=0)
print(df.groupby('fold').agg(files=('path', 'size'), originals=('is_orig', 'sum')))
print('classes present in every fold val:', (per > 0).all(axis=0).sum(), '/ 100')
print('min originals of a class in a fold val:', per.values.min())
