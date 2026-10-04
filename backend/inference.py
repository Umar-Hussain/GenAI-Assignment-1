import time
import io
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, ImageFilter

def load_model(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])

def preprocess_image(image_bytes: bytes, size=(128, 128), to_tanh=False):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = img.size
    img_resized = img.resize(size, Image.Resampling.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    if to_tanh:
        img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0), orig_size

def postprocess_image(output_array: np.ndarray, target_size=None, is_sketch=False) -> Image.Image:
    arr = np.squeeze(output_array, axis=0)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))

    if is_sketch:
        # Convert [-1, 1] or raw output to 0-255
        if arr.min() < 0.0:
            arr = (arr * 0.5) + 0.5
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.squeeze(arr, axis=2)
        sk_img = Image.fromarray(arr, mode="L")

        # Dynamic range stretching to ensure bold, deep black lines and full coverage
        sk_img = ImageOps.autocontrast(sk_img, cutoff=1)

        if target_size and max(target_size) > 128:
            display_size = (max(256, min(target_size[0], 512)), max(256, min(target_size[1], 512)))
            sk_img = sk_img.resize(display_size, Image.Resampling.LANCZOS)
        return sk_img

    # Standard Restoration (RGB)
    if arr.min() < 0.0:
        arr = (arr * 0.5) + 0.5
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=2)
        res_img = Image.fromarray(arr, mode="L")
    else:
        res_img = Image.fromarray(arr, mode="RGB")

    # High-quality Lanczos upsampling to native upload resolution (or min 256) + mild unsharp mask
    if target_size and max(target_size) > 128:
        display_size = (max(256, min(target_size[0], 512)), max(256, min(target_size[1], 512)))
        res_img = res_img.resize(display_size, Image.Resampling.LANCZOS)
        res_img = res_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    elif res_img.size == (128, 128):
        # Default crisp display size for modern browser containers
        res_img = res_img.resize((256, 256), Image.Resampling.LANCZOS)
        res_img = res_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=2))

    return res_img

def run_inference(session: ort.InferenceSession, inputs):
    if not isinstance(inputs, dict):
        input_name = session.get_inputs()[0].name
        feed_dict = {input_name: inputs}
    else:
        feed_dict = inputs

    start_time = time.time()
    outputs = session.run(None, feed_dict)
    inference_time = (time.time() - start_time) * 1000.0
    return outputs, inference_time
