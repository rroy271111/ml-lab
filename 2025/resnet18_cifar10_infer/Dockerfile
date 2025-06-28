FROM python:3.10-slim

RUN pip install torch torchvision flask pillow

WORKDIR /app

COPY scripts/app.py .
COPY resnet18_cifar10.pth .

CMD ["python", "app.py"]
