import torch
from pytorch_pretrained_gans import make_gan
from pytorch_pretrained_biggan.utils import one_hot_from_names, truncated_noise_sample
import numpy as np
import matplotlib.pyplot as plt

# Definir tus 10 clases específicas
class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 
               'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

# Crear vector one-hot desde los nombres de clase
print('hola')

class_vec = one_hot_from_names(class_names, batch_size=len(class_names))
print('hola')

# Obtener índices numéricos de las clases de ImageNet
imagenet_classes = {
    'tench': 0,
    'English springer': 217,
    'cassette player': 482,
    'chain saw': 491,
    'church': 497,
    'French horn': 566,
    'garbage truck': 569,
    'gas pump': 571,
    'golf ball': 574,
    'parachute': 701
}


def generar_imagenes_gan(modelo_gan, modelName, class_vec, truncation=0.4):
    """
    Genera y muestra imágenes condicionales usando un modelo GAN preentrenado.

    Args:
        modelo_gan: Modelo GAN preentrenado (e.g. BigGAN, SAGAN).
        class_names: Lista de nombres de clases para generar imágenes.
        truncation: Factor de truncamiento del ruido (default 0.4).

    Returns:
        output: Tensor con imágenes generadas (batch_size, 3, H, W)
    """

    # Crear vector de ruido truncado
    noise_vec = truncated_noise_sample(truncation=truncation, batch_size=len(class_names))

    # Convertir vectores a tensores de PyTorch
    class_vec = torch.from_numpy(class_vec)
    noise_vec = torch.from_numpy(noise_vec)

    # Mover al dispositivo (GPU si disponible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_gan.to(device)
    class_vec = class_vec.to(device)
    noise_vec = noise_vec.to(device)

    # Generar imágenes condicionales
    with torch.no_grad():
        output = modelo_gan(noise_vec, class_vec)

    # Mostrar imágenes generadas
    output_np = output.cpu().numpy()
    output_np = (output_np + 1) / 2  # Normalizar de [-1,1] a [0,1]

    fig, axes = plt.subplots(2, len(class_names)//2, figsize=(15, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = np.transpose(output_np[i], (1, 2, 0))  # (H,W,3)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(class_names[i])
        ax.axis('off')

    plt.title(modelName)
    plt.tight_layout()
    plt.show()

    return output

# Crear tensor de índices de clases
class_indices = torch.tensor([imagenet_classes[name] for name in class_names], dtype=torch.long)


def generar_imagenes_studiogan(modelo_gan, modelName, class_indices, truncation=0.4):
    """
    Genera y muestra imágenes condicionales usando un modelo StudioGAN preentrenado.

    Args:
        modelo_gan: Modelo StudioGAN preentrenado (e.g. SAGAN, ContraGAN).
        modelName: Nombre del modelo para título.
        class_indices: Tensor con índices numéricos de clases ImageNet.
        truncation: Factor de truncamiento del ruido (default 0.4).

    Returns:
        output: Tensor con imágenes generadas (batch_size, 3, H, W)
    """

    # Crear vector de ruido truncado
    noise_vec = truncated_noise_sample(truncation=truncation, batch_size=len(class_indices))

    # Convertir vectores a tensores de PyTorch
    noise_vec = torch.from_numpy(noise_vec).float()

    # Mover al dispositivo (GPU si disponible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_gan.to(device)
    class_indices = class_indices.to(device)
    noise_vec = noise_vec.to(device)

    # Generar imágenes condicionales
    with torch.no_grad():
        output = modelo_gan(noise_vec, class_indices)

    # Mostrar imágenes generadas
    output_np = output.cpu().numpy()
    output_np = (output_np + 1) / 2  # Normalizar de [-1,1] a [0,1]

    fig, axes = plt.subplots(2, len(class_indices)//2, figsize=(15, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = np.transpose(output_np[i], (1, 2, 0))  # (H,W,3)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(class_names[i])
        ax.axis('off')

    plt.suptitle(modelName, fontsize=16)
    plt.tight_layout()
    plt.show()

    return output

G_BigGan = make_gan(gan_type='biggan')  # BigGAN preentrenado (256px)
generar_imagenes_gan(G_BigGan, 'BigGan', class_vec)

# Ejemplo de uso correcto:
G = make_gan(gan_type='studiogan', model_name='SAGAN')  # asegúrate que el modelo es condicional
generar_imagenes_studiogan(G, 'SAGAN StudioGAN', class_indices)

# Ejemplo de uso correcto:
G = make_gan(gan_type='studiogan', model_name='SNGAN')  # asegúrate que el modelo es condicional
generar_imagenes_studiogan(G, 'SAGAN StudioGAN', class_indices) 

