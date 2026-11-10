import os
import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from util import util

# scikit-image 0.17 still ships the old compare_* names; prefer the new ones.
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_psnr as psnr
    from skimage.measure import compare_ssim as ssim


def window_starts(total, patch, overlap):
    """Tile origins along one axis, always including the last full window.

    overlap must be strictly smaller than patch: at overlap == patch the stride would
    collapse to 1 and the window count would explode (silently, by orders of magnitude),
    so that is rejected rather than clamped.
    """
    if not 0 <= overlap < patch:
        raise ValueError('overlap must satisfy 0 <= overlap < patch '
                         '(got overlap=%d, patch=%d)' % (overlap, patch))
    step = patch - overlap
    pos = list(range(0, max(1, total - patch + 1), step))
    if pos[-1] != total - patch:
        pos.append(total - patch)
    return pos


def sliding_inference(net, vol, patch, overlap, device):
    """Run the generator over a full volume in patches and reassemble it.

    `vol` is (1, 1, Z, H, W) in [-1, 1]. The learned positional encoding ties the
    network to one patch size, so inference must use the training patch size.

    Any axis shorter than the patch is replicate-padded temporarily; the padding is
    cropped off before returning, so the result has exactly the input dimensions.
    Overlapping predictions are averaged (sum / count).
    """
    pz, py, px = patch
    if not 0 <= overlap < min(pz, py, px):
        raise ValueError('--overlap must satisfy 0 <= overlap < min(patch) = %d '
                         '(got overlap=%d, patch=%s). An overlap equal to the patch '
                         'would reduce the stride to 1 voxel.'
                         % (min(pz, py, px), overlap, str(patch)))
    Z, H, W = vol.shape[2], vol.shape[3], vol.shape[4]
    pad = (0, max(0, px - W), 0, max(0, py - H), 0, max(0, pz - Z))
    if any(pad):
        vol = F.pad(vol, pad, mode='replicate')
    Zp, Hp, Wp = vol.shape[2], vol.shape[3], vol.shape[4]

    acc = torch.zeros(1, 1, Zp, Hp, Wp)
    cnt = torch.zeros(1, 1, Zp, Hp, Wp)
    n = 0
    for z0 in window_starts(Zp, pz, overlap):
        for y0 in window_starts(Hp, py, overlap):
            for x0 in window_starts(Wp, px, overlap):
                chunk = vol[:, :, z0:z0 + pz, y0:y0 + py, x0:x0 + px].to(device)
                with torch.no_grad():
                    out = net(chunk).cpu()
                acc[:, :, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += out
                cnt[:, :, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += 1.0
                n += 1
    out = acc / cnt.clamp(min=1.0)
    return out[:, :, :Z, :H, :W], n


def to01(t):
    """[-1,1] tensor -> (Z,H,W) float array in [0,1], as train.py de-normalises."""
    return np.clip(t.numpy()[0, 0] * 0.5 + 0.5, 0.0, 1.0)


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    vol_dir = os.path.join(web_dir, 'volumes')
    if not os.path.exists(vol_dir):
        os.makedirs(vol_dir)

    patch = (opt.fineSizeZ, opt.fineSize, opt.fineSize)
    net = model.netG
    net.eval()
    device = next(net.parameters()).device
    print('sliding-window inference: patch %s, overlap %d, device %s'
          % (str(patch), opt.overlap, device))

    rows = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        real_A, real_B = data['A'], data['B']
        case = os.path.basename(data['A_paths'][0])[:-len('_X.tif')]

        fake_B, n_win = sliding_inference(net, real_A, patch, opt.overlap, device)
        assert fake_B.shape == real_A.shape, \
            'reconstruction %s != input %s' % (tuple(fake_B.shape), tuple(real_A.shape))

        f01, r01 = to01(fake_B), to01(real_B)
        # Metrics are computed in the normalised space the model is trained in, with a
        # FIXED data range. They therefore never depend on target statistics that would
        # be unavailable at inference time. (train.py's per-volume-max PSNR is fine for
        # ranking epochs but is not comparable between volumes.)
        mae = float(np.abs(f01 - r01).mean())
        val_psnr = float(psnr(r01, f01, data_range=1.0))
        val_ssim = float(ssim(r01, f01, data_range=1.0))
        rows.append((case, real_A.shape[2], mae, val_psnr, val_ssim))
        print('%04d: %s  shape=%s  windows=%d  MAE=%.5f  PSNR=%.3f  SSIM=%.4f'
              % (i, case, str(tuple(real_A.shape[2:])), n_win, mae, val_psnr, val_ssim))

        # predicted and reference volumes, uint16 on the same normalised scale as *_Y.tif
        tifffile.imwrite(os.path.join(vol_dir, case + '_fake_B.tif'),
                         (f01 * 65535.0).astype(np.uint16))
        tifffile.imwrite(os.path.join(vol_dir, case + '_real_B.tif'),
                         (r01 * 65535.0).astype(np.uint16))

        # central-slice PNG gallery, unchanged machinery (tensor2im takes the mid-Z slice)
        visuals = OrderedDict([('real_A', util.tensor2im(real_A)),
                               ('fake_B', util.tensor2im(fake_B)),
                               ('real_B', util.tensor2im(real_B))])
        visualizer.save_images(webpage, visuals, [data['A_paths'][0]],
                               aspect_ratio=opt.aspect_ratio)

    webpage.save()

    with open(os.path.join(web_dir, 'metrics.csv'), 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['case', 'native_z', 'mae', 'psnr', 'ssim'])
        for r in rows:
            w.writerow(r)
        if rows:
            arr = np.array([[r[2], r[3], r[4]] for r in rows])
            w.writerow(['MEAN', '', arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()])
            w.writerow(['STD', '', arr[:, 0].std(), arr[:, 1].std(), arr[:, 2].std()])
    if rows:
        arr = np.array([[r[2], r[3], r[4]] for r in rows])
        print('--- %d volumes: MAE %.5f+-%.5f  PSNR %.3f+-%.3f  SSIM %.4f+-%.4f ---'
              % (len(rows), arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean(),
                 arr[:, 1].std(), arr[:, 2].mean(), arr[:, 2].std()))
    print('volumes written to: %s' % vol_dir)
