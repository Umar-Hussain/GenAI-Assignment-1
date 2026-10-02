import time
import io
import numpy as np
import onnxruntime as ort
from PIL import Image

def load_model(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])

def preprocess_image(image_bytes: bytes, size=(128, 128), to_tanh=False) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size, Image.Resampling.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    if to_tanh:
        img_np = (img_np - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0)

def postprocess_image(output_array: np.ndarray) -> Image.Image:
    arr = np.squeeze(output_array, axis=0)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.min() < 0.0:
        arr = (arr * 0.5) + 0.5
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=2)
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")

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
