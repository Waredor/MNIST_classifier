import os
import json
import numpy as np
import torch
from utils.models import MnistMLP
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

with open(file='configs/pipeline_config.json', mode='r', encoding='utf-8') as f:
    model_config = json.load(f)

used_model_filename = model_config["in_app_used_model_filename"]

# Загрузка модели
model_filepath = os.path.join('models', used_model_filename)
model = MnistMLP()
if os.path.exists(model_filepath):
    state_dict = torch.load(model_filepath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

else:
    raise FileNotFoundError(f"Неверный путь к файлу {used_model_filename}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 1, 28, 28)

            img_tensor = torch.tensor(img_array, dtype=torch.float32)

            with torch.no_grad():
                pred = model(img_tensor)
                prediction = int(torch.argmax(pred, dim=1).item())

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)