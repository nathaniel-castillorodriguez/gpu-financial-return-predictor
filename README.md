# gpu-financial-return-predictor
GPU-accelerated S&amp;P 500 return prediction engine built with PyTorch and custom CUDA kernel. Achieves 26.69x speedup over CPU (302.75s → 11.34s) using fused batch norm + LeakyReLU kernel. Deployed to GCP Cloud Run. Live API: https://return-predictor-654964924336.us-central1.run.app
