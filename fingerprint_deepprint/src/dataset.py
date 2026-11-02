"""
Fingerprint Dataset Module for SOCOFing and FVC Datasets.

Implements metadata parsing, subject-independent data splitting (preventing data leakage),
and PyTorch Dataset / DataLoader creation.
"""

import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .preprocessing import FingerprintPreprocessor


def parse_socofing_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parses SOCOFing filename into subject_id, gender, hand, finger_type, and alteration.

    Examples:
        Real: '40__F_Right_middle_finger.BMP' -> subject: 40, finger_id: '40_Right_middle'
        Altered: '580__M_Right_ring_finger_CR.BMP' -> subject: 580, finger_id: '580_Right_ring'
    """
    basename = os.path.basename(filename)
    # Pattern for real or altered
    pattern = r'^(\d+)__([MF])_([A-Za-z]+)_([a-z]+)_finger(?:_([A-Za-z0-9]+))?\.BMP$'
    match = re.match(pattern, basename, re.IGNORECASE)
    if not match:
        return None

    subject_id, gender, hand, finger_type, alteration = match.groups()
    finger_name = f"{hand}_{finger_type}"
    unique_finger_id = f"{subject_id}_{finger_name}"

    return {
        "file_path": filename,
        "filename": basename,
        "subject_id": int(subject_id),
        "gender": gender,
        "hand": hand,
        "finger_type": finger_type,
        "finger_name": finger_name,
        "unique_finger_id": unique_finger_id,
        "alteration": alteration if alteration else "Real"
    }


class SOCOFingDataset(Dataset):
    """
    PyTorch Dataset for SOCOFing images.
    """
    def __init__(self, records: List[Dict], transform=None, label_map: Optional[Dict[str, int]] = None):
        """
        Args:
            records: List of parsed file metadata dictionaries.
            transform: PyTorch transforms to apply to the PIL Image.
            label_map: Mapping from unique_finger_id (or subject_id) to integer class index.
        """
        self.records = records
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        image_path = rec["file_path"]

        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')

        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        label = -1
        if self.label_map is not None:
            label = self.label_map.get(rec["unique_finger_id"], -1)

        return tensor_img, label, rec["unique_finger_id"]


def load_socofing_records(
    socofing_dir: str,
    include_altered: bool = True,
    alteration_levels: Optional[List[str]] = None
) -> List[Dict]:
    """
    Loads all image paths and parsed metadata from a SOCOFing directory.

    Expected structure:
        socofing_dir/
            Real/
            Altered/ (Altered-Easy, Altered-Medium, Altered-Hard)
    """
    records = []

    # Real impressions
    real_dir = os.path.join(socofing_dir, "Real")
    if os.path.isdir(real_dir):
        for f in glob.glob(os.path.join(real_dir, "*.BMP")):
            meta = parse_socofing_filename(f)
            if meta:
                records.append(meta)

    # Altered impressions
    if include_altered:
        altered_base = os.path.join(socofing_dir, "Altered")
        levels = alteration_levels or ["Altered-Easy", "Altered-Medium", "Altered-Hard"]
        for lvl in levels:
            lvl_dir = os.path.join(altered_base, lvl)
            if os.path.isdir(lvl_dir):
                for f in glob.glob(os.path.join(lvl_dir, "*.BMP")):
                    meta = parse_socofing_filename(f)
                    if meta:
                        records.append(meta)

    return records


def create_subject_independent_splits(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:
    """
    CRITICAL: Prevents data leakage by splitting at the SUBJECT level.
    Fingerprints belonging to subjects in the test set NEVER appear in the training or validation sets.

    Returns:
        train_records, val_records, test_records, train_label_map
    """
    import random
    rng = random.Random(seed)

    # Group records by subject
    subjects = sorted(list(set(r["subject_id"] for r in records)))
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train + n_val])
    test_subs = set(subjects[n_train + n_val:])

    train_records = [r for r in records if r["subject_id"] in train_subs]
    val_records = [r for r in records if r["subject_id"] in val_subs]
    test_records = [r for r in records if r["subject_id"] in test_subs]

    # Build classification label map for training fingers
    train_finger_ids = sorted(list(set(r["unique_finger_id"] for r in train_records)))
    train_label_map = {fid: idx for idx, fid in enumerate(train_finger_ids)}

    return train_records, val_records, test_records, train_label_map


def build_gallery_and_probe_sets(
    records: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Constructs 1:1 and 1:N evaluation sets from test records.
    Gallery: Pristine Real impression of each finger (1 image per finger).
    Probe: Altered impressions or secondary impressions of the same finger.
    """
    gallery = []
    probe = []

    for r in records:
        if r["alteration"] == "Real":
            gallery.append(r)
        else:
            probe.append(r)

    # Edge case: if a finger has no Real image, fallback to first available
    gallery_fids = set(r["unique_finger_id"] for r in gallery)
    for r in records:
        if r["unique_finger_id"] not in gallery_fids:
            gallery.append(r)
            gallery_fids.add(r["unique_finger_id"])

    return gallery, probe


class FVC2004Dataset(Dataset):
    """
    PyTorch Dataset for FVC2004 DB1_B.
    Filenames are formatted as: <subject>_<impression>.tif (e.g., '101_1.tif' to '101_8.tif').
    """
    def __init__(self, fvc_dir: str, transform=None):
        self.fvc_dir = fvc_dir
        self.transform = transform
        self.records = []

        pattern = os.path.join(fvc_dir, "*.tif")
        for f in sorted(glob.glob(pattern)):
            base = os.path.basename(f)
            parts = base.replace(".tif", "").split("_")
            if len(parts) == 2:
                sub_id, imp_id = parts
                self.records.append({
                    "file_path": f,
                    "filename": base,
                    "finger_id": sub_id,
                    "impression_id": int(imp_id)
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        rec = self.records[idx]
        img = Image.open(rec["file_path"])
        if img.mode != 'L':
            img = img.convert('L')
        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)
        return tensor_img, rec["finger_id"], rec["impression_id"]

