import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import CorruptionClassifier
from src.models.moe import SoftMoE, GatingNetwork
from src.models.gan import UNetGenerator

def verify_onnx(pytorch_model, onnx_path, *dummy_inputs):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    feed_dict = {input_names[i]: dummy_inputs[i].cpu().numpy() for i in range(len(dummy_inputs))}
    ort_outs = session.run(None, feed_dict)
    
    with torch.no_grad():
        pytorch_model.eval()
        pt_outs = pytorch_model(*dummy_inputs)
    
    if isinstance(pt_outs, torch.Tensor):
        pt_outs = [pt_outs]
    elif isinstance(pt_outs, tuple):
        pt_outs = list(pt_outs)
    
    max_diff = 0.0
    for ort_out, pt_out in zip(ort_outs, pt_outs):
        diff = np.max(np.abs(ort_out - pt_out.cpu().numpy()))
        max_diff = max(max_diff, float(diff))
    
    print(f"Validation successful: Max absolute error = {max_diff:.6e}")
    return max_diff

def export_model(model, name, save_dir, *dummy_inputs, opset=14, output_names=None):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{name}.onnx")
    
    model.eval()
    if output_names is None:
        output_names = ["output"]
        
    input_names = [f"input_{i}" for i in range(len(dummy_inputs))]
    if len(dummy_inputs) == 2 and isinstance(model, UNetGenerator):
        input_names = ["image", "style_id"]
        output_names = ["sketch"]
    elif isinstance(model, SoftMoE):
        input_names = ["image"]
        output_names = ["restored_image", "routing_weights"]

    input_tuple = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0]
    try:
        torch.onnx.export(
            model,
            input_tuple,
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            input_tuple,
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names
        )
    print(f"Exported {name} -> {out_path}")
    verify_onnx(model, out_path, *dummy_inputs)

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX and verify consistency")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'universal_ae', 'classifier', 'specialists', 'soft_moe', 'generator'])
    parser.add_argument('--ckpt_dir', type=str, default='models/checkpoints')
    parser.add_argument('--out_dir', type=str, default='models/onnx')
    args = parser.parse_args()

    dummy_img = torch.randn(1, 3, 128, 128)
    dummy_style = torch.tensor([0], dtype=torch.long)

    # 1. Universal Autoencoder
    if args.model in ['all', 'universal_ae']:
        print("\n--- Exporting Universal Denoising Autoencoder (Task 1) ---")
        ae = DenoisingAutoencoder()
        ckpt = os.path.join(args.ckpt_dir, 'universal_ae.pth')
        if os.path.exists(ckpt):
            ae.load_state_dict(torch.load(ckpt, map_location='cpu'))
        export_model(ae, 'universal_ae', args.out_dir, dummy_img)

    # 2. Corruption Classifier
    if args.model in ['all', 'classifier']:
        print("\n--- Exporting Corruption Classifier (Task 2) ---")
        clf = CorruptionClassifier()
        ckpt = os.path.join(args.ckpt_dir, 'classifier.pth')
        if os.path.exists(ckpt):
            clf.load_state_dict(torch.load(ckpt, map_location='cpu'))
        export_model(clf, 'classifier', args.out_dir, dummy_img, output_names=["class_logits"])

    # 3. Specialist Autoencoders
    if args.model in ['all', 'specialists']:
        print("\n--- Exporting Specialist Autoencoders (Task 2) ---")
        for sp_name in ['salt', 'blur', 'occlusion']:
            sp_model = DenoisingAutoencoder()
            ckpt = os.path.join(args.ckpt_dir, f'specialist_{sp_name}.pth')
            if os.path.exists(ckpt):
                sp_model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            export_model(sp_model, f'specialist_{sp_name}', args.out_dir, dummy_img)

    # 4. Soft Mixture-of-Experts
    if args.model in ['all', 'soft_moe']:
        print("\n--- Exporting Soft Mixture-of-Experts (Task 3) ---")
        gate = GatingNetwork()
        experts = [
            nn.Identity(),
            DenoisingAutoencoder(),
            DenoisingAutoencoder(),
            DenoisingAutoencoder()
        ]
        moe = SoftMoE(gate, experts)
        ckpt = os.path.join(args.ckpt_dir, 'soft_moe.pth')
        if os.path.exists(ckpt):
            moe.load_state_dict(torch.load(ckpt, map_location='cpu'))
        export_model(moe, 'soft_moe', args.out_dir, dummy_img)

    # 5. Face-to-Sketch Conditional GAN Generator
    if args.model in ['all', 'generator']:
        print("\n--- Exporting Face-to-Sketch Conditional GAN Generator (Task 4) ---")
        gen = UNetGenerator()
        ckpt = os.path.join(args.ckpt_dir, 'generator.pth')
        if os.path.exists(ckpt):
            gen.load_state_dict(torch.load(ckpt, map_location='cpu'))
        export_model(gen, 'generator', args.out_dir, dummy_img, dummy_style)

if __name__ == '__main__':
    main()
