# GenAI Assignment #1: Multi-Task Image Restoration & Conditional Style Synthesis

Unified Generative AI suite covering:
1. **Task 1: Universal Multi-Corruption Denoising Autoencoder** (bottleneck compression for clean, salt-and-pepper, Gaussian blur, and rectangular occlusion).
2. **Task 2: Corruption Classification & Hard Routing** (4-class CNN router + dedicated specialist autoencoders with clean identity bypass).
3. **Task 3: Jointly Trained Soft Mixture-of-Experts (MoE)** (continuous differentiable routing weights with temperature scaling and entropy balance regularization).
4. **Task 4: Style-Conditioned Face-to-Sketch Synthesis** (U-Net Generator + PatchGAN Discriminator with categorical style embeddings).
5. **Additional Features**: Interactive Before/After Split Comparison Slider, Real-time Webcam capture & synthesis, Residual Error Heatmap generator, and System Latency/Throughput Diagnostic Telemetry.

---

## Directory Layout
```text
genai-assignment/
├── backend/                  # FastAPI REST backend
│   ├── main.py               # API routes (/api/universal-restore, hard-route, soft-moe, face-to-sketch, corrupt)
│   ├── inference.py          # ONNX Runtime graph execution and pre/post-processing
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # React + Tailwind CSS client
│   ├── src/
│   │   ├── components/       # Workspace cards, sliders, webcam modal, benchmarks
│   │   ├── utils/api.js      # REST API client
│   │   ├── App.jsx           # Main workspace coordinator
│   │   └── main.jsx          # Vite React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── checkpoints/          # PyTorch trained model weights (.pth)
│   └── onnx/                 # Exported, validated ONNX models (.onnx)
├── src/
│   ├── data/                 # Dataset loader, corruption pipeline & manifests
│   ├── models/               # PyTorch models (Autoencoder, Classifier, SoftMoE, GAN)
│   ├── train/                # Training pipelines with Optuna HPO & metrics logging
│   ├── export_onnx.py        # ONNX export and verification script
│   └── evaluate.py           # Evaluation script (PSNR, SSIM, L1, Confusion Matrix)
├── docker/                   # Dockerfile.backend & Dockerfile.frontend
├── docker-compose.yml        # One-command orchestration
├── report/                   # IEEE research technical paper (LaTeX & BibTeX)
├── scripts/smoke_test.py     # Automated end-to-end API smoke test
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows: venv\Scripts\activate, Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Export / Validate ONNX Inference Models
Export PyTorch models to ONNX and verify numerical consistency against PyTorch outputs:
```bash
python -m src.export_onnx
```

### 3. Run Application via Docker Compose (Recommended)
Launch both the backend and frontend in isolated containers with a single command:
```bash
docker-compose up --build
```
- **Web Interface**: `http://localhost:3000`
- **FastAPI Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 4. Local Execution Without Docker (Alternative)
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

---

## Model Training & Evaluation

```bash
# Prepare dataset and deterministic validation/test manifests
python -m src.data.prepare_data

# Train Task 1: Universal Autoencoder (with Optuna)
python -m src.train.train_autoencoder --data_dir data/ --epochs 20

# Train Task 2: Classifier and Specialists
python -m src.train.train_classifier --data_dir data/ --epochs 15
python -m src.train.train_specialists --data_dir data/ --epochs 20

# Train Task 3: Soft Mixture-of-Experts
python -m src.train.train_moe --data_dir data/ --epochs 20

# Train Task 4: Face-to-Sketch Conditional GAN
python -m src.train.train_gan --data_dir data/ --epochs 30

# Run Full Evaluation Suite (generates results/evaluation_metrics.json)
python -m src.evaluate
```

---

## API Endpoints

| Method | Endpoint | Parameters | Description |
|---|---|---|---|
| `GET` | `/health` | None | Returns system health & available ONNX models |
| `POST` | `/api/universal-restore` | `file` (multipart) | Task 1: Reconstructs clean image via universal AE |
| `POST` | `/api/hard-route` | `file` (multipart) | Task 2: Classifies corruption & routes to specialist |
| `POST` | `/api/soft-moe` | `file` (multipart) | Task 3: Soft MoE restoration with 4 routing weights |
| `POST` | `/api/face-to-sketch` | `file`, `style` (1, 2, or 3) | Task 4: Conditional GAN facial sketch synthesis |
| `POST` | `/api/corrupt` | `file`, `corruption_type`, `params` | Programmatic runtime corruption generator |

---

## Automated Smoke Test
Verify backend endpoints and model inference:
```bash
python scripts/smoke_test.py
```
