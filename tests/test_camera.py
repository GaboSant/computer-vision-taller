import numpy as np
from PIL import Image

def imagen(n: int):
    if n == 1: return Image.open(r'.\data\veneno-roadster.jpg')
    elif n == 2: return Image.open(r'.\data\Cat.jpg')

def test_radial_distortion(n=1, k1=0.0, k2=0.0):
    from cvtools.camera import apply_radial_distortion
    import matplotlib.pyplot as plt

    imagen_color = np.array(imagen(n))
    imagen_distorsionada = apply_radial_distortion(imagen_color, k1=k1, k2=k2)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_distorsionada)
    plt.title(f'Imagen con Distorsión Radial (k1={k1}, k2={k2})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_focal_distortion(n=1, f=500):
    from cvtools.camera import apply_focal_distortion
    import matplotlib.pyplot as plt

    imagen_color = np.array(imagen(n))
    imagen_distorsionada = apply_focal_distortion(imagen_color, f)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_distorsionada)
    plt.title(f'Imagen con Distorsión Focal (f={f})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()