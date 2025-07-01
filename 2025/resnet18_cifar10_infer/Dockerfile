# Base image
FROM python:3.10-slim

# Install dependencies
RUN pip install torch torchvision flask pillow

# Copy app
WORKDIR /app
COPY app.py .
COPY resnet18_cifar10.pth .

# Run app
CMD ["python", "app.py"]