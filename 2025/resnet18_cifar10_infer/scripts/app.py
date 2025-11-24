# Flask API for Inference

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))  
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))

])

cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'     
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_name = cifar10_classes[predicted.item()]
    return jsonify({'prediction': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    