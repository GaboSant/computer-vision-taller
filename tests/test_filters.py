import numpy as np
from cvtools import filters
from PIL import Image

def imagen(n: int):
    if n == 1: return Image.open(r'.\data\veneno-roadster.jpg')
    elif n == 2: return Image.open(r'.\data\Cat.jpg')

def test_convolucion(n=1, kernel=None):
    import matplotlib.pyplot as plt

    if kernel is None:
        # Kernel de ejemplo: Filtro de desenfoque
        kernel = np.array([[0, 2, 0],
                           [2, 8, 2],
                           [0, 2, 0]], dtype=np.float32)

    imagen_color = imagen(n)
    imagen_convolucionada = filters.convolucion(imagen_color, kernel)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_convolucionada)
    plt.title('Imagen con Convolución')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_sobel(n=1):
    import matplotlib.pyplot as plt

    imagen_color = imagen(n)
    imagen_sobel_x = filters.sobel_x(imagen_color)
    imagen_sobel_y = filters.sobel_y(imagen_color)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imagen_sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imagen_sobel_y, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_canny(n=1, umbral_bajo=50, umbral_alto=150):
    import matplotlib.pyplot as plt

    imagen_color = imagen(n)
    imagen_canny = filters.canny(imagen_color, umbral_bajo, umbral_alto)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_canny, cmap='gray')
    plt.title('Detector de Canny')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_laplacian(n=1):
    import matplotlib.pyplot as plt

    imagen_color = imagen(n)
    imagen_laplaciana = filters.filtro_laplaciano(imagen_color)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_color)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_laplaciana, cmap='gray')
    plt.title('Filtro Laplaciano')
    plt.axis('off')

    plt.tight_layout()
    plt.show()