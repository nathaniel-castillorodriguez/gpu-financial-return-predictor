
# GPU-Accelerated Financial Return Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.0-green)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-blue)

## Live API
**https://return-predictor-654964924336.us-central1.run.app**

## Overview
GPU-accelerated LSTM neural network that predicts next-day S&P 500 returns using 10 years of historical data across 50 tickers. Built with a custom CUDA kernel that fuses batch normalization and LeakyReLU into a single GPU pass, eliminating memory round-trips and achieving 26.69x speedup over CPU baseline.

## Benchmark Results

| Method | Training Time | Throughput | Speedup |
|---|---|---|---|
| CPU | 302.75s | 3,270 samples/sec | 1x |
| GPU (standard) | 12.44s | 79,564 samples/sec | 24.33x |
| GPU + Custom Kernel | 11.34s | 87,302 samples/sec | 26.69x |

## Architecture
- **Model:** 2-layer LSTM, hidden size 64, dropout 0.2 — 51,265 parameters
- **Input:** 20-day sequences of 4 engineered features per ticker
- **Features:** 5-day rolling return, 20-day volatility, RSI, volume z-score
- **Target:** Next-day return prediction
- **Data:** 50 S&P 500 tickers, 2014–2024, 124,750 total samples

## Custom CUDA Kernel
Wrote a fused batch normalization + LeakyReLU CUDA kernel that combines two separate GPU operations into one pass. This eliminates an intermediate memory round-trip between operations — the same kernel fusion principle used in NVIDIA's cuDNN and FlashAttention.

Kernel verification: **max difference from PyTorch baseline = 0.00e+00 (exact match)**

## Results

### Training Loss Curves + Throughput
![Charts](results/charts.png)

### Predicted vs Actual Returns
![Predicted vs Actual](results/predicted_vs_actual.png)

## Deployment
Containerized with Docker and deployed to GCP Cloud Run.

Live endpoint returns benchmark data and accepts prediction requests:
```bash
curl https://return-predictor-654964924336.us-central1.run.app
```

## Tech Stack
- Python, PyTorch, CUDA/C++
- Docker, Kubernetes
- GCP Cloud Run
- yfinance, scikit-learn, matplotlib

## Project Structure
```
├── app.py                    # Flask API serving predictions
├── Dockerfile                # Container configuration
├── deployment.yaml           # Kubernetes deployment config
├── requirements.txt          # Dependencies
├── kernel.cu                 # Custom fused CUDA kernel
└── results/
    ├── charts.png            # Loss curves + throughput comparison
    └── predicted_vs_actual.png
```
