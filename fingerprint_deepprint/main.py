"""
Main Execution Script for DeepPrint-Inspired Fingerprint Recognition.

Allows training, embedding extraction, and complete evaluation from the command line.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.preprocessing import FingerprintPreprocessor
from src.dataset import (
    load_socofing_records,
    create_subject_independent_splits,
    build_gallery_and_probe_sets,
    SOCOFingDataset,
    FVC2004Dataset
)
from src.model import DeepPrintResNet, DeepPrintMobileNet
from src.train import Trainer
from src.embeddings import EmbeddingExtractor, compress_embedding, decompress_embedding
from src.matching import FingerprintMatcher
from src.evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser(description="DeepPrint-Inspired Fingerprint Recognition")
    parser.add_argument("--data_dir", type=str, default="../data/archive/SOCOFing",
                        help="Path to SOCOFing root folder (containing Real and Altered)")
    parser.add_argument("--fvc_dir", type=str, default="../data/fvc2004/DB1_B",
                        help="Optional path to FVC2004 DB1_B folder for cross-dataset evaluation")
    parser.add_argument("--model_arch", type=str, choices=["resnet50", "mobilenet_v3"], default="resnet50",
                        help="Pretrained CNN backbone architecture")
    parser.add_argument("--embedding_dim", type=int, default=192,
                        help="Fixed-length representation dimension (default: 192)")
    parser.add_argument("--epochs_s1", type=int, default=3,
                        help="Number of epochs for Stage 1 (warmup, frozen backbone)")
    parser.add_argument("--epochs_s2", type=int, default=5,
                        help="Number of epochs for Stage 2 (fine-tuning, unfrozen backbone)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and eval")
    parser.add_argument("--lr_s1", type=float, default=1e-3, help="Learning rate for Stage 1")
    parser.add_argument("--lr_s2", type=float, default=1e-4, help="Learning rate for Stage 2")
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="Subsample subjects for rapid experimentation on CPU (0 = all subjects)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and run evaluation using pre-saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to checkpoint for evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    print(f"[*] Selected Backbone: {args.model_arch}")
    print(f"[*] Fixed Embedding Dimension: {args.embedding_dim}")

    # 1. Dataset Loading and Splitting
    print(f"[*] Loading SOCOFing dataset from: {args.data_dir}")
    all_records = load_socofing_records(args.data_dir, include_altered=True, alteration_levels=["Altered-Easy"])
    print(f"[*] Total parsed records: {len(all_records)}")

    if args.max_subjects > 0:
        allowed_subs = set(list(set(r["subject_id"] for r in all_records))[:args.max_subjects])
        all_records = [r for r in all_records if r["subject_id"] in allowed_subs]
        print(f"[*] Subsampled to {args.max_subjects} subjects ({len(all_records)} images)")

    train_rec, val_rec, test_rec, train_label_map = create_subject_independent_splits(
        all_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    print(f"[*] Subject-Independent Split (No Data Leakage):")
    print(f"    Train images: {len(train_rec)} (Classes: {len(train_label_map)})")
    print(f"    Val images:   {len(val_rec)}")
    print(f"    Test images:  {len(test_rec)}")

    # 2. Preprocessor & DataLoaders
    preprocessor = FingerprintPreprocessor(img_size=224, use_clahe=True)
    train_tf = preprocessor.get_train_transforms()
    eval_tf = preprocessor.get_eval_transforms()

    train_dataset = SOCOFingDataset(train_rec, transform=train_tf, label_map=train_label_map)
    val_dataset = SOCOFingDataset(val_rec, transform=eval_tf, label_map=train_label_map)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model Instantiation
    num_classes = len(train_label_map)
    if args.model_arch == "resnet50":
        model = DeepPrintResNet(num_classes=num_classes, embedding_dim=args.embedding_dim, pretrained=True)
    else:
        model = DeepPrintMobileNet(num_classes=num_classes, embedding_dim=args.embedding_dim, pretrained=True)

    # 4. Training
    if not args.eval_only:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir="checkpoints",
            results_dir="results"
        )
        trainer.fit(
            epochs_stage1=args.epochs_s1,
            epochs_stage2=args.epochs_s2,
            lr_stage1=args.lr_s1,
            lr_stage2=args.lr_s2
        )
    else:
        if os.path.exists(args.checkpoint):
            print(f"[*] Loading weights from: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Running with initial pretrained weights.")

    # 5. Fixed-Length Embedding Extraction on Held-Out Test Set
    print("\n[*] Extracting embeddings on held-out test subjects...")
    gallery_rec, probe_rec = build_gallery_and_probe_sets(test_rec)
    print(f"    Gallery (Enrolled Pristine): {len(gallery_rec)}")
    print(f"    Probe (Queries/Altered):    {len(probe_rec)}")

    gallery_dataset = SOCOFingDataset(gallery_rec, transform=eval_tf)
    probe_dataset = SOCOFingDataset(probe_rec, transform=eval_tf)

    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False)
    probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False)

    extractor = EmbeddingExtractor(model=model, device=device)
    gallery_embs, gallery_fids = extractor.extract_dataset(gallery_loader)
    probe_embs, probe_fids = extractor.extract_dataset(probe_loader)

    print(f"    Gallery embeddings shape: {gallery_embs.shape}")
    print(f"    Probe embeddings shape:   {probe_embs.shape}")

    # 6. Test 200-Byte Compression Routine
    sample_compressed = compress_embedding(gallery_embs[0])
    sample_decompressed = decompress_embedding(sample_compressed)
    recon_error = np.mean(np.abs(gallery_embs[0] - sample_decompressed))
    print(f"[*] Template Compression Test:")
    print(f"    Uncompressed: 192 float32 = 768 bytes")
    print(f"    Compressed:   Exact footprint = {len(sample_compressed)} bytes (192 uint8 + 8 bytes floats)")
    print(f"    Decompression Mean Abs Error: {recon_error:.6f}")

    # 7. Biometric Evaluation
    evaluator = Evaluator(results_dir="results")
    print("\n[*] Calculating Genuine & Impostor Cosine Similarity Scores...")
    gen_scores, imp_scores = evaluator.compute_scores(
        probe_embs, probe_fids, gallery_embs, gallery_fids
    )
    print(f"    Genuine comparisons:  {len(gen_scores)}")
    print(f"    Impostor comparisons: {len(imp_scores)}")

    evaluator.plot_score_distributions(gen_scores, imp_scores)

    roc_dict = evaluator.compute_roc_and_eer(gen_scores, imp_scores)
    tar_01, th_01 = evaluator.compute_tar_at_far(gen_scores, imp_scores, target_far=0.001)
    tar_001, th_001 = evaluator.compute_tar_at_far(gen_scores, imp_scores, target_far=0.0001)

    evaluator.plot_roc_curve(roc_dict, tar_01, tar_001)

    id_metrics = evaluator.compute_identification_metrics(
        probe_embs, probe_fids, gallery_embs, gallery_fids, max_rank=50
    )
    evaluator.plot_cmc_curve(id_metrics["cmc_curve"])

    print("\n" + "="*50)
    print("BIOMETRIC EVALUATION SUMMARY (DEEPPRINT BENCHMARKS)")
    print("="*50)
    print(f"Rank-1 Identification Accuracy:    {id_metrics['rank_1']:.2f}%")
    print(f"Hit Rate @ Penetration Rate 1%:    {id_metrics['hr_at_pr_1']:.2f}% (Top {id_metrics['pr_1_pct_k']} candidates)")
    print(f"Equal Error Rate (EER):            {roc_dict['eer']*100.0:.2f}% (Threshold: {roc_dict['eer_threshold']:.4f})")
    print(f"TAR @ FAR = 0.1% (10^-3):          {tar_01*100.0:.2f}% (Threshold: {th_01:.4f})")
    print(f"TAR @ FAR = 0.01% (10^-4):         {tar_001*100.0:.2f}% (Threshold: {th_001:.4f})")
    print("="*50)
    print("[*] All evaluation figures saved to 'results/' directory.\n")


if __name__ == "__main__":
    main()

