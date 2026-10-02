import io
import time
import requests
from PIL import Image
import numpy as np

def run_smoke_test(base_url="http://127.0.0.1:8000"):
    print(f"Connecting to backend at {base_url}...")
    
    # 1. Health check
    res = requests.get(f"{base_url}/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("Health check PASSED:", res.json())

    # Create dummy 128x128 image
    img = Image.new('RGB', (128, 128), color=(120, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    # 2. Corrupt test
    res = requests.post(
        f"{base_url}/api/corrupt",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"corruption_type": "salt_pepper", "params": '{"prob": 0.05}'}
    )
    assert res.status_code == 200, f"Corrupt failed: {res.text}"
    print("Corrupt endpoint PASSED")

    # 3. Universal Restore
    res = requests.post(
        f"{base_url}/api/universal-restore",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200, f"Universal restore failed: {res.text}"
    data = res.json()
    print(f"Universal restore PASSED (latency: {data['inference_time_ms']} ms)")

    # 4. Hard Route
    res = requests.post(
        f"{base_url}/api/hard-route",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200, f"Hard route failed: {res.text}"
    data = res.json()
    print(f"Hard route PASSED (selected expert: {data['selected_expert']}, latency: {data['inference_time_ms']} ms)")

    # 5. Soft MoE
    res = requests.post(
        f"{base_url}/api/soft-moe",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200, f"Soft MoE failed: {res.text}"
    data = res.json()
    print(f"Soft MoE PASSED (weights: {[round(w, 3) for w in data['routing_weights']]}, latency: {data['inference_time_ms']} ms)")

    # 6. Face to Sketch
    res = requests.post(
        f"{base_url}/api/face-to-sketch",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"style": 1}
    )
    assert res.status_code == 200, f"Face to sketch failed: {res.text}"
    data = res.json()
    print(f"Face to sketch PASSED (latency: {data['inference_time_ms']} ms)")

    print("\nALL 6 ENDPOINTS PASSED SMOKE TEST!")

if __name__ == '__main__':
    run_smoke_test()
