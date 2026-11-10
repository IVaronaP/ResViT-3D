"""Build the 3D ResViT dataset from the original Allen Institute CZI volumes.

brightfield volume (X)  ->  fibrillarin fluorescence volume (Y)

Differences from the 2D pipeline, and why:

  * Target channel is the one named exactly 'EGFP' (fibrillarin-mEGFP). The 2D
    export_czi.py used `ch_label = C - 2`, which is the channel named 'H3342' --
    the Hoechst DNA stain, not fibrillarin. Channels are therefore selected BY NAME,
    which is also robust to the '_1'/'_2' suffix variation across the 80 files.
  * The full native Z range is preserved (60-75 slices here). Nothing is cropped,
    padded, resized or interpolated; training extracts 3D patches instead.
  * Output is uint16 rather than 8-bit PNG, keeping ~8x more target precision.

The stored uint16 is NOT raw camera ADU: 0 maps to that volume's p_lo percentile and
65535 to its p_hi. The (lo, hi) pair for every volume and channel is written to
norm_params.csv so the mapping is reproducible and auditable.

Originals are opened read-only and never modified.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import tifffile


def open_czi(path):
    """Open a CZI and return (reader, channel names, (z,y,x) spacing in microns).

    Only metadata is touched here; pixels are read per channel by read_channel, so the
    five channels we do not use are never pulled off disk.

    Tries bioio (what export_czi.py uses) then aicsimageio. One of the two must be
    installed, together with its CZI backend.
    """
    try:
        from bioio import BioImage as Reader
    except ImportError:
        from aicsimageio import AICSImage as Reader
    im = Reader(path)
    try:
        names = list(im.channel_names)
    except Exception:
        names = list(im.get_channel_names())
    try:
        sp = im.physical_pixel_sizes
        spacing = (sp.Z, sp.Y, sp.X)
    except Exception:
        spacing = (None, None, None)
    return im, names, spacing


def read_channel(im, c):
    """Read ONE channel as (Z, Y, X)."""
    return im.get_image_data("ZYX", T=0, C=c)


def pick_channels(names):
    """Input = brightfield, target = fibrillarin-mEGFP, chosen by channel name."""
    tgt = [i for i, n in enumerate(names) if n.strip().upper() == 'EGFP']
    inp = [i for i, n in enumerate(names) if n.strip().upper().startswith('BRIGHT')]
    if not tgt:
        raise ValueError("no channel named exactly 'EGFP' in %s" % names)
    if not inp:
        raise ValueError("no channel starting with 'Bright' in %s" % names)
    return inp[-1], tgt[-1]


def normalize_u16(vol, p_lo, p_hi):
    """Per-volume percentile clip and rescale to uint16. Returns (vol, lo, hi)."""
    lo, hi = np.percentile(vol, (p_lo, p_hi))
    if hi <= lo:
        return np.zeros(vol.shape, np.uint16), float(lo), float(hi)
    out = np.clip(vol, lo, hi).astype(np.float32)
    out = (out - lo) / (hi - lo)
    return (out * 65535.0).astype(np.uint16), float(lo), float(hi)


def make_splits(stems, train_fraction, val_fraction, seed):
    """Split per source volume, mirroring split_by_czi_folder.py (same seed/logic).

    A volume appears in exactly one split, so no patch or slice can leak between them.
    """
    cases = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_train = int(round(len(cases) * train_fraction))
    train_all, test = cases[:n_train], cases[n_train:]

    rng2 = random.Random(seed)
    tr = list(train_all)
    rng2.shuffle(tr)
    n_val = int(round(len(tr) * val_fraction))
    val, train = tr[:n_val], tr[n_val:]
    return sorted(train), sorted(val), sorted(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--czi_dir', required=True, help='directory holding the original *.czi')
    ap.add_argument('--out_dir', required=True, help='destination for train/ val/ test/')
    ap.add_argument('--p_lo', type=float, default=0.5)
    ap.add_argument('--p_hi', type=float, default=99.5)
    ap.add_argument('--train_fraction', type=float, default=0.8)
    ap.add_argument('--val_fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N volumes (0 = all); for smoke tests')
    ap.add_argument('--dry_run', action='store_true',
                    help='inspect and report channels/shapes, write nothing')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.czi_dir) if f.lower().endswith('.czi'))
    if not files:
        raise SystemExit('no .czi found in %s' % args.czi_dir)
    stems = [os.path.splitext(f)[0] for f in files]
    train, val, test = make_splits(stems, args.train_fraction, args.val_fraction, args.seed)
    split_of = {}
    for s in train:
        split_of[s] = 'train'
    for s in val:
        split_of[s] = 'val'
    for s in test:
        split_of[s] = 'test'
    print('split per source volume: %d train / %d val / %d test' % (len(train), len(val), len(test)))

    todo = files[:args.limit] if args.limit else files
    if args.limit:
        print('LIMIT: processing only the first %d of %d volumes' % (len(todo), len(files)))
    if args.dry_run:
        print('DRY RUN: nothing will be written')
    else:
        for sp in ('train', 'val', 'test'):
            d = os.path.join(args.out_dir, sp)
            if not os.path.exists(d):
                os.makedirs(d)

    norm_rows, spacings = [], {}
    for k, fname in enumerate(todo):
        stem = os.path.splitext(fname)[0]
        path = os.path.join(args.czi_dir, fname)
        im, names, spacing = open_czi(path)
        ci, ct = pick_channels(names)
        y_raw = read_channel(im, ct)
        print('[%3d/%3d] %s' % (k + 1, len(todo), stem))
        print('          channels=%s' % (names,))
        print('          input  = C%d (%s)   target = C%d (%s)' % (ci, names[ci], ct, names[ct]))
        print('          shape=%s dtype=%s spacing(z,y,x)=%s split=%s'
              % (y_raw.shape, y_raw.dtype, spacing, split_of[stem]))
        # Sanity check that the target really is EGFP. Measured over the WHOLE volume
        # (not a single in-focus slice, which gives a higher figure): EGFP ~3.3,
        # H3342 ~1.2, empty duplicate channels ~1.05.
        p50, p995 = np.percentile(y_raw, (50, 99.5))
        print('          target p99.5/p50 = %.2f  (EGFP ~3.3, H3342 ~1.2)'
              % (p995 / max(p50, 1e-6)))

        if args.dry_run:
            continue

        x_raw = read_channel(im, ci)
        x, xlo, xhi = normalize_u16(x_raw, args.p_lo, args.p_hi)
        y, ylo, yhi = normalize_u16(y_raw, args.p_lo, args.p_hi)
        assert x.shape == y.shape
        out = os.path.join(args.out_dir, split_of[stem])
        tifffile.imwrite(os.path.join(out, stem + '_X.tif'), x)
        tifffile.imwrite(os.path.join(out, stem + '_Y.tif'), y)
        norm_rows.append((stem, split_of[stem], 'X', names[ci], xlo, xhi, x.shape[0]))
        norm_rows.append((stem, split_of[stem], 'Y', names[ct], ylo, yhi, y.shape[0]))
        spacings[stem] = spacing

    if args.dry_run:
        return

    with open(os.path.join(args.out_dir, 'norm_params.csv'), 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['case', 'split', 'role', 'channel_name', 'p_lo_adu', 'p_hi_adu', 'native_z'])
        for r in norm_rows:
            w.writerow(r)
    with open(os.path.join(args.out_dir, 'split_manifest.csv'), 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['case', 'split'])
        for s in sorted(split_of):
            w.writerow([s, split_of[s]])
    with open(os.path.join(args.out_dir, 'voxel_spacing.json'), 'w') as fh:
        json.dump({'spacing_zyx_um': spacings,
                   'p_lo': args.p_lo, 'p_hi': args.p_hi, 'seed': args.seed},
                  fh, indent=2)
    print('done -> %s' % args.out_dir)


if __name__ == '__main__':
    main()
