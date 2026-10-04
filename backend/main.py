import io
import base64
import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.inference import load_model, preprocess_image, postprocess_image, run_inference
except ImportError:
    from inference import load_model, preprocess_image, postprocess_image, run_inference

app = FastAPI(
    title="GenAI Multi-Task Restoration & Synthesis API",
    description="Backend service for Denoising Autoencoder, Hard Routing, Soft MoE, and Conditional GAN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/onnx"))
models = {}

def get_model(name: str):
    if name not in models:
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"ONNX model '{name}' not found at {path}. Please run export_onnx.py first.")
        models[name] = load_model(path)
    return models[name]

def encode_image(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/health")
def health_check():
    loaded_models = list(models.keys())
    available_files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
    return {
        "status": "ok",
        "loaded_models": loaded_models,
        "available_onnx_models": [f for f in available_files if f.endswith(".onnx")]
    }

@app.post("/api/universal-restore")
async def universal_restore(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_array, orig_size = preprocess_image(image_bytes, to_tanh=False)
    
    session = get_model("universal_ae.onnx")
    outputs, inference_time = run_inference(session, input_array)
    
    restored_img = postprocess_image(outputs[0], target_size=orig_size)
    return {
        "restored_image": encode_image(restored_img),
        "inference_time_ms": round(inference_time, 2)
    }

@app.post("/api/hard-route")
async def hard_route(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_array, orig_size = preprocess_image(image_bytes, to_tanh=False)
    
    classifier_session = get_model("classifier.onnx")
    cls_outputs, cls_time = run_inference(classifier_session, input_array)
    
    logits = cls_outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = (exp_logits / np.sum(exp_logits)).tolist()
    expert_idx = int(np.argmax(probs))
    
    specialist_map = {
        0: "identity_bypass",
        1: "specialist_salt.onnx",
        2: "specialist_blur.onnx",
        3: "specialist_occlusion.onnx"
    }

    if expert_idx == 0:
        clean_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_dim = (max(256, min(clean_img.size[0], 512)), max(256, min(clean_img.size[1], 512)))
        clean_img = clean_img.resize(target_dim, Image.Resampling.LANCZOS)
        return {
            "restored_image": encode_image(clean_img),
            "classifier_probs": probs,
            "selected_expert": expert_idx,
            "expert_name": "Identity Bypass (Clean)",
            "inference_time_ms": round(cls_time, 2)
        }
    
    expert_file = specialist_map[expert_idx]
    expert_session = get_model(expert_file)
    exp_outputs, exp_time = run_inference(expert_session, input_array)
    
    restored_img = postprocess_image(exp_outputs[0], target_size=orig_size)
    return {
        "restored_image": encode_image(restored_img),
        "classifier_probs": probs,
        "selected_expert": expert_idx,
        "expert_name": expert_file,
        "inference_time_ms": round(cls_time + exp_time, 2)
    }

@app.post("/api/soft-moe")
async def soft_moe(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_array, orig_size = preprocess_image(image_bytes, to_tanh=False)
    
    session = get_model("soft_moe.onnx")
    outputs, inference_time = run_inference(session, input_array)
    
    restored_img = postprocess_image(outputs[0], target_size=orig_size)
    
    if len(outputs) > 1:
        weights = outputs[1][0].tolist()
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    
    return {
        "restored_image": encode_image(restored_img),
        "routing_weights": weights,
        "inference_time_ms": round(inference_time, 2)
    }

@app.post("/api/face-to-sketch")
async def face_to_sketch(file: UploadFile = File(...), style: int = Form(1)):
    image_bytes = await file.read()
    input_array, orig_size = preprocess_image(image_bytes, to_tanh=True)
    
    style_idx = np.array([max(0, min(2, style - 1))], dtype=np.int64)
    session = get_model("generator.onnx")
    
    inputs = {
        session.get_inputs()[0].name: input_array,
        session.get_inputs()[1].name: style_idx
    }
    outputs, inference_time = run_inference(session, inputs)
    
    sketch_img = postprocess_image(outputs[0], target_size=orig_size, is_sketch=True)
    return {
        "sketch_image": encode_image(sketch_img),
        "style_selected": style,
        "inference_time_ms": round(inference_time, 2)
    }

@app.post("/api/corrupt")
async def corrupt_image(file: UploadFile = File(...), corruption_type: str = Form(...), params: str = Form("{}")):
    image_bytes = await file.read()
    orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = orig_img.size
    
    img = orig_img.resize((128, 128), Image.Resampling.BILINEAR)
    p = json.loads(params) if params else {}

    if corruption_type == "clean":
        pass
    elif corruption_type == "salt_pepper":
        prob = float(p.get("prob", random.uniform(0.02, 0.15)))
        arr = np.array(img).copy()
        mask = np.random.rand(*arr.shape[:2])
        salt = mask < (prob / 2.0)
        pepper = (mask >= (prob / 2.0)) & (mask < prob)
        arr[salt] = [255, 255, 255]
        arr[pepper] = [0, 0, 0]
        img = Image.fromarray(arr)
    elif corruption_type == "gaussian_blur":
        kernel = int(p.get("kernel_size", random.choice([3, 5, 7])))
        sigma = float(p.get("sigma", random.uniform(0.5, 2.5)))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif corruption_type == "occlusion":
        num_rects = int(p.get("num_rects", random.randint(1, 3)))
        coverage = float(p.get("coverage", random.uniform(0.10, 0.35)))
        arr = np.array(img).copy()
        h, w, _ = arr.shape
        per_rect_area = int((h * w * coverage) / num_rects)
        for _ in range(num_rects):
            rw = random.randint(max(1, int(per_rect_area ** 0.5) // 2), min(w, int(per_rect_area ** 0.5) * 2))
            rh = max(1, per_rect_area // rw)
            rw, rh = min(rw, w), min(rh, h)
            x = random.randint(0, w - rw)
            y = random.randint(0, h - rh)
            arr[y:y+rh, x:x+rw] = [0, 0, 0]
        img = Image.fromarray(arr)

    target_dim = (max(256, min(orig_size[0], 512)), max(256, min(orig_size[1], 512)))
    img_display = img.resize(target_dim, Image.Resampling.LANCZOS)

    return {
        "corrupted_image": encode_image(img_display),
        "corruption_type": corruption_type
    }
