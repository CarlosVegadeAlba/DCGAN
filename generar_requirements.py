import importlib
import importlib.metadata
import sys

# Lista de librerías que usas
libraries = [
    "os",
    "re",
    "gc",
    "random",
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "json",
    "time",
    "copy",
    "IPython",
    "Pillow",
    "cv2"
]

# Mapeo para paquetes cuyo nombre de instalación es diferente al nombre de importación
package_name_map = {
    "cv2": "opencv-python",
    "Pillow": "Pillow",
    "IPython": "ipython",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision"
}

# Solo paquetes que se pueden instalar con pip (excluye stdlib como os, time, etc.)
installable = [lib for lib in libraries if lib in package_name_map]

requirements = []

for lib in installable:
    package_name = package_name_map[lib]
    try:
        version = importlib.metadata.version(package_name)
        requirements.append(f"{package_name}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"[!] {package_name} no está instalado. Lo omito.", file=sys.stderr)

# Escribir requirements.txt
with open("requirements.txt", "w") as f:
    for line in requirements:
        f.write(line + "\n")

print("requirements.txt generado con éxito.")
