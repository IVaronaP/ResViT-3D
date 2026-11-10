import os.path
import random
import torch
import numpy as np
import tifffile
from data.base_dataset import BaseDataset


class AlignedDataset(BaseDataset):
    """Aligned 3D volume pairs: <stem>_X.tif (brightfield) and <stem>_Y.tif (fluorescence).

    Replaces the 2D side-by-side PNG loader. Volumes are stored at their native depth
    (60-75 slices here), so patches are extracted on the fly; nothing on disk is resized,
    cropped or interpolated.

    Three behaviours, matching the 2D original's intent:
      train : `patches_per_volume` random patches, random flip
      val   : `val_patches_per_volume` fixed patches, no augmentation (deterministic,
              so the validation metric is comparable from epoch to epoch)
      test  : the complete volume at its native size, no crop, no augmentation
              (test.py tiles it and reassembles the original dimensions)
    """

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # Capture the phase ONCE, here. train.py sets opt.phase='val' when it validates
        # and never restores it, so reading opt.phase inside __getitem__ would silently
        # turn the *training* set into a validation set after the first validated epoch.
        self.phase = opt.phase

        names = sorted(n for n in os.listdir(self.dir_AB) if n.endswith('_X.tif'))
        self.AB_paths = [os.path.join(self.dir_AB, n[:-len('_X.tif')]) for n in names]
        assert len(self.AB_paths) > 0, 'no *_X.tif volumes found in %s' % self.dir_AB
        assert opt.input_nc == 1 and opt.output_nc == 1, \
            'the 3D loader stores one channel per volume (input_nc=output_nc=1)'

        if self.phase == 'train':
            self.n_per_vol = opt.patches_per_volume
        elif self.phase == 'val':
            self.n_per_vol = opt.val_patches_per_volume
            # Fixed validation patch positions, drawn once here and reused every epoch.
            # Stored as fractions of the valid offset range so they are independent of
            # each volume's depth and need no header read at startup.
            self.val_frac = []
            for vi in range(len(self.AB_paths)):
                rng = random.Random(1234 + vi)
                for _ in range(self.n_per_vol):
                    self.val_frac.append((rng.random(), rng.random(), rng.random()))
        else:
            self.n_per_vol = 1

    def pad_depth(self, vol, dz):
        """Replicate-pad Z up to dz. In memory only; the stored volume is untouched.

        Only needed when a volume is shallower than the patch (e.g. a 60-slice stack with
        fineSizeZ=64). With the default fineSizeZ=48 this never triggers.
        """
        if vol.shape[0] >= dz:
            return vol
        before = (dz - vol.shape[0]) // 2
        after = dz - vol.shape[0] - before
        return np.pad(vol, ((before, after), (0, 0), (0, 0)), mode='edge')

    def to_tensor(self, vol):
        """uint16 -> (1, Z, H, W) float in [-1, 1].

        The stored uint16 holds the per-volume percentile-normalised intensity, so
        /65535 then (x-0.5)/0.5 reproduces the 2D ToTensor() + Normalize(0.5, 0.5).
        """
        t = torch.from_numpy(vol.astype(np.float32) / 65535.0)
        return ((t - 0.5) / 0.5).unsqueeze(0)

    def __getitem__(self, index):
        vi = index // self.n_per_vol
        stem = self.AB_paths[vi]
        A_path, B_path = stem + '_X.tif', stem + '_Y.tif'
        A = tifffile.imread(A_path)
        B = tifffile.imread(B_path)
        assert A.shape == B.shape, \
            'input/target shape mismatch for %s: %s vs %s' % (stem, A.shape, B.shape)

        if self.phase == 'test':
            # full volume, native size; test.py does the sliding window
            return {'A': self.to_tensor(A), 'B': self.to_tensor(B),
                    'A_paths': A_path, 'B_paths': B_path}

        dz, dy, dx = self.opt.fineSizeZ, self.opt.fineSize, self.opt.fineSize
        A, B = self.pad_depth(A, dz), self.pad_depth(B, dz)
        assert A.shape[1] >= dy and A.shape[2] >= dx, \
            'volume %s is smaller than the XY patch (%s < %d)' % (stem, A.shape[1:], dy)

        # one origin, applied to both volumes, so input and target stay aligned
        if self.phase == 'train':
            z0 = random.randint(0, A.shape[0] - dz)
            y0 = random.randint(0, A.shape[1] - dy)
            x0 = random.randint(0, A.shape[2] - dx)
        else:
            fz, fy, fx = self.val_frac[index]
            z0 = int(fz * (A.shape[0] - dz))
            y0 = int(fy * (A.shape[1] - dy))
            x0 = int(fx * (A.shape[2] - dx))

        A = A[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
        B = B[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
        A, B = self.to_tensor(A), self.to_tensor(B)

        # random flip along X, as the 2D version flipped width (train only)
        if self.phase == 'train' and (not self.opt.no_flip) and random.random() < 0.5:
            idx = torch.LongTensor([i for i in range(A.size(3) - 1, -1, -1)])
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.AB_paths) * self.n_per_vol

    def name(self):
        return 'AlignedDataset'
