import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
import nltk
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Just a try to see if loading the model works

nltk.download('wordnet')
# 1. Cargar el modelo BigGAN preentrenado (256px)
model = BigGAN.from_pretrained('biggan-deep-256')  

# 2. Preparar un vector de clase (one-hot) y vector de ruido aleatorio
class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 
               'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
class_vec  = one_hot_from_names(class_names, batch_size=len(class_names))  # 10 clases
noise_vec  = truncated_noise_sample(truncation=0.4, batch_size=len(class_names))

# 3. Convertir a tensores de PyTorch
class_vec  = torch.from_numpy(class_vec)
noise_vec  = torch.from_numpy(noise_vec)

# (Opcional: mover a GPU si disponible)
model.to('cuda'); class_vec = class_vec.to('cuda'); noise_vec = noise_vec.to('cuda')

# 4. Generar un batch de imágenes sintéticas
with torch.no_grad():
    output = model(noise_vec, class_vec, truncation=0.4)  # tensor [10, 3, 256, 256]

# Mostrar todas las imágenes generadas
for i, class_name in enumerate(class_names):
    img = output[i].cpu().numpy()  # extraer imagen i del batch

    # Transformar de [-1,1] a [0,255]
    img = np.clip(((img + 1) / 2.0) * 255, 0, 255).astype(np.uint8)

    # Convertir tensor a formato imagen (C,H,W) => (H,W,C)
    img = np.transpose(img, (1, 2, 0))

    # Mostrar la imagen generada
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Clase generada: {class_name}')
    plt.show()


