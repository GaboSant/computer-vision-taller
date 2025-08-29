import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cvtools import color, plotting

def imagen(n: int):
    if n == 1: return Image.open(r'.\data\veneno-roadster.jpg')
    elif n == 2: return Image.open(r'.\data\Cat.jpg')

def test_rgb(n=1, plot=False):
    imagen_color = imagen(n)
    
    if plot:
        plotting.mostrar_canales(imagen_color.split(), "rgb")
    else:
        for canal in imagen_color.split():
            print(np.array(canal), "\n")    

def test_rgb_to_any(espacio: str, n=1, plot=False):
    imagen_color = imagen(n)
    match espacio.lower():
        case 'hsv':
            color.rgb_a_hsv(imagen_color, plot)
        case 'lab':
            color.rgb_a_lab(imagen_color, plot)
        case 'yuv':
            color.rgb_a_yuv(imagen_color, plot)

def test_histogram(n=1):
    imagen_color = imagen(n)
    color.histograma_colores(imagen_color)

def test_cuantizacion(n=1, k=[16, 64]):
    imagen_color = imagen(n)

    if len(k) < 3:
        # Creamos una figura con 1 fila y (len(k) + 1) columnas
        fig, axes = plt.subplots(1, len(k) + 1, figsize=(5 * (len(k) + 1), 5))
        
        # Mostrar la imagen original
        axes[0].imshow(imagen_color)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Mostrar las cuantizadas
        for i, k_val in enumerate(k, start=1):
            imagen_cuantizada = color.cuantizacion_simple(imagen_color, k_val)
            axes[i].imshow(imagen_cuantizada)
            axes[i].set_title(f"Cuantizada k={k_val}")
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
    
    else:
        # Caso original: se muestran de a una
        for k_val in k:
            imagen_cuantizada = color.cuantizacion_simple(imagen_color, k_val)
            plt.figure(figsize=(8, 8))
            plt.imshow(imagen_cuantizada)
            plt.title(f"Imagen Cuantizada con k={k_val}")
            plt.axis("off")
            plt.show()

import matplotlib.pyplot as plt

def test_cuantizacion_con_tamano(n=1, k=[16, 64]):
    imagen_color = imagen(n)  # suponiendo que devuelve PIL.Image
    imagen_original, tamano_original = color.reducir_peso(imagen_color, 256)  # 256 ≈ sin pérdida fuerte

    if len(k) < 3:
        # Subplots: original + todas las cuantizadas
        fig, axes = plt.subplots(1, len(k) + 1, figsize=(5 * (len(k) + 1), 5))

        # Imagen original
        axes[0].imshow(imagen_original)
        axes[0].set_title(f"Original\n{tamano_original:.1f} KB")
        axes[0].axis("off")

        # Cuantizadas
        for i, k_val in enumerate(k, start=1):
            imagen_cuantizada, tamano_cuantizada = color.reducir_peso(imagen_color, k_val)

            axes[i].imshow(imagen_cuantizada)
            axes[i].set_title(f"k={k_val}\n{tamano_cuantizada:.1f} KB")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    else:
        # Mostrar de a una
        for k_val in k:
            imagen_cuantizada, tamano_cuantizada = color.reducir_peso(imagen_color, k_val)

            plt.figure(figsize=(8, 8))
            plt.imshow(imagen_cuantizada)
            plt.title(f"Cuantizada k={k_val} ({tamano_cuantizada:.1f} KB)")
            plt.axis("off")
            plt.show()

