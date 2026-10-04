import os
import sys
import argparse
import subprocess
import torch
import numpy as np
from PIL import Image

def run_cmd(cmd_list):
    print(f"\n[RUNNING] {' '.join(cmd_list)}", flush=True)
    res = subprocess.run([sys.executable] + cmd_list, check=True)
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="End-to-end training, ONNX export, and verification pipeline")
    parser.add_argument("--skip_ae", action="store_true", help="Skip universal autoencoder training")
    parser.add_argument("--skip_clf", action="store_true", help="Skip classifier training")
    parser.add_argument("--skip_spec", action="store_true", help="Skip specialists training")
    parser.add_argument("--skip_moe", action="store_true", help="Skip MoE training")
    parser.add_argument("--skip_gan", action="store_true", help="Skip GAN training")
    args = parser.parse_args()

    ckpt_dir = "models/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Universal Autoencoder
    uae_path = os.path.join(ckpt_dir, "universal_ae.pth")
    if not args.skip_ae and not os.path.exists(uae_path):
        print("\n=== STEP 1: Training Universal Autoencoder ===", flush=True)
        run_cmd(["-m", "src.train.train_autoencoder", "--epochs", "5", "--batch_size", "32"])
    else:
        print(f"Universal AE checkpoint exists: {uae_path}", flush=True)

    # 2. Classifier
    clf_path = os.path.join(ckpt_dir, "classifier.pth")
    if not args.skip_clf and not os.path.exists(clf_path):
        print("\n=== STEP 2: Training Corruption Classifier ===", flush=True)
        run_cmd(["-m", "src.train.train_classifier", "--epochs", "5", "--batch_size", "32"])
    else:
        print(f"Classifier checkpoint exists: {clf_path}", flush=True)

    # 3. Specialists
    spec_paths = [os.path.join(ckpt_dir, f"specialist_{s}.pth") for s in ["salt", "blur", "occlusion"]]
    if not args.skip_spec and not all(os.path.exists(p) for p in spec_paths):
        print("\n=== STEP 3: Training Specialist Autoencoders ===", flush=True)
        run_cmd(["-m", "src.train.train_specialists", "--epochs", "4", "--batch_size", "32"])
    else:
        print("All specialist checkpoints exist.", flush=True)

    # 4. Soft MoE
    moe_path = os.path.join(ckpt_dir, "soft_moe.pth")
    if not args.skip_moe and not os.path.exists(moe_path):
        print("\n=== STEP 4: Fine-Tuning Soft Mixture-of-Experts ===", flush=True)
        run_cmd(["-m", "src.train.train_moe", "--epochs", "3", "--batch_size", "32"])
    else:
        print(f"Soft MoE checkpoint exists: {moe_path}", flush=True)

    # 5. Conditional GAN
    gan_path = os.path.join(ckpt_dir, "generator.pth")
    if not args.skip_gan and not os.path.exists(gan_path):
        print("\n=== STEP 5: Training Face-to-Sketch Conditional GAN ===", flush=True)
        run_cmd(["-m", "src.train.train_gan", "--epochs", "4", "--batch_size", "16"])
    else:
        print(f"GAN generator checkpoint exists: {gan_path}", flush=True)

    # 6. Export all models to ONNX
    print("\n=== STEP 6: Exporting all models to ONNX ===", flush=True)
    run_cmd(["-m", "src.export_onnx"])

    # 7. Evaluate and verify
    print("\n=== STEP 7: Generating Evaluation Metrics ===", flush=True)
    run_cmd(["-m", "src.evaluate"])

    print("\n[SUCCESS] End-to-end training and export completed successfully!", flush=True)

if __name__ == "__main__":
    main()
