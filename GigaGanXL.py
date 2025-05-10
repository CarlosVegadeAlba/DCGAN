import torch
import legacy
import dnnlib
import requests
import os
from tqdm import tqdm
import time

weights_url = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
weights_path = "imagenet256.pkl"

# Función para descargar pesos si no existen
def download_weights(url, path):
    if not os.path.exists(path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                file.write(data)
    else:
        print(f"{path} ya existe. Se omite la descarga.")

# Función para cargar el modelo correctamente desde el pickle
def load_styleganxl(path, device='cpu'):
    with dnnlib.util.open_url(path) as f:
        model = legacy.load_network_pkl(f)['G_ema'].to(device)
    return model

if __name__ == "__main__":
    device = torch.device('cpu')  # Usando CPU para evitar problemas de compilación CUDA

    download_weights(weights_url, weights_path)
    G = load_styleganxl(weights_path, device=device)
    G.eval()

    batch_size = 50

    # Generar vectores latentes y etiquetas aleatorias (por ejemplo, clase 0)
    z = torch.randn(batch_size, G.z_dim).to(device)
    c = torch.zeros(batch_size, G.c_dim).to(device)

    # Medir tiempo de generación del batch
    start_time = time.time()

    with torch.no_grad():
        img_batch = G(z, c)

    elapsed_time = time.time() - start_time

    print(f"Batch generado en {elapsed_time:.2f} segundos.")
    print(f"Forma del tensor generado: {img_batch.shape}")
