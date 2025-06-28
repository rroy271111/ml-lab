# ResNet18 CIFAR10 Inference (2025)

This project performs inference on CIFAR10 using a pretrained ResNet18 model.

## Structure
- scripts/ — application code (Flask/FastAPI)
- notebooks/ — exploration & batch inference
- images/ — sample images for testing
- data/ — CIFAR10 batch files (excluded tar.gz)

## Usage
```
docker build -t resnet18-cifar10 .
docker run -p 8000:8000 resnet18-cifar10
```
