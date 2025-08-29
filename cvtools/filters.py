import numpy as np
from PIL import Image, ImageFilter

def convolucion(imagen: Image.Image, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica una convolución genérica a una imagen usando un kernel dado.

    Parámetros
    ----------
    imagen : PIL.Image
        Imagen de entrada (RGB o escala de grises).
    kernel : np.ndarray
        Matriz del kernel de convolución (debe ser 2D).

    Retorna
    -------
    np.ndarray
        Imagen resultante después de aplicar la convolución.
    """
    # Convertir imagen a NumPy
    img_array = np.array(imagen, dtype=np.float32)

    # Si es en escala de grises -> agregar dimensión
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]

    # Tamaño del kernel
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Padding
    padded = np.pad(img_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')

    # Crear salida
    salida = np.zeros_like(img_array)

    # Convolución canal por canal
    for c in range(img_array.shape[2]):
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                region = padded[i:i + k_h, j:j + k_w, c]
                salida[i, j, c] = np.sum(region * kernel)

    # Normalizar al rango válido [0,255]
    salida = np.clip(salida, 0, 255).astype(np.uint8)

    # Quitar canal si era gris
    if salida.shape[2] == 1:
        salida = salida[:, :, 0]

    return salida

# --- Definir kernels de Sobel ---
KERNEL_SOBEL_X = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

KERNEL_SOBEL_Y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)


def sobel_x(imagen: Image.Image) -> np.ndarray:
    """
    Aplica el filtro Sobel en la dirección X.
    """
    return convolucion(imagen.convert("L"), KERNEL_SOBEL_X)


def sobel_y(imagen: Image.Image) -> np.ndarray:
    """
    Aplica el filtro Sobel en la dirección Y.
    """
    return convolucion(imagen.convert("L"), KERNEL_SOBEL_Y)

def canny(imagen: Image.Image, umbral_bajo: int = 50, umbral_alto: int = 150) -> np.ndarray:
    """
    Aplica el detector de bordes de Canny a una imagen en escala de grises.

    Parámetros:
    -----------
    imagen : PIL.Image
        Imagen de entrada (RGB o escala de grises).
    umbral_bajo : int
        Umbral bajo para histéresis.
    umbral_alto : int
        Umbral alto para histéresis.

    Retorna:
    --------
    np.ndarray
        Imagen binaria con los bordes detectados.
    """
    # 1. Convertir a escala de grises
    gris = imagen.convert("L")
    img = np.array(gris, dtype=np.float32)

    # 2. Suavizado con filtro Gaussiano
    img = np.array(gris.filter(ImageFilter.GaussianBlur(radius=1.4)), dtype=np.float32)

    # 3. Gradientes Sobel
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)

    Gx = convolucion(gris, Kx).astype(np.float32)
    Gy = convolucion(gris, Ky).astype(np.float32)

    magnitud = np.hypot(Gx, Gy)
    magnitud = magnitud / magnitud.max() * 255
    angulo = np.arctan2(Gy, Gx)

    # 4. Supresión no máxima
    M, N = magnitud.shape
    Z = np.zeros((M, N), dtype=np.float32)
    ang = angulo * 180. / np.pi
    ang[ang < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # Dirección 0
            if (0 <= ang[i,j] < 22.5) or (157.5 <= ang[i,j] <= 180):
                q = magnitud[i, j+1]
                r = magnitud[i, j-1]
            # Dirección 45
            elif (22.5 <= ang[i,j] < 67.5):
                q = magnitud[i+1, j-1]
                r = magnitud[i-1, j+1]
            # Dirección 90
            elif (67.5 <= ang[i,j] < 112.5):
                q = magnitud[i+1, j]
                r = magnitud[i-1, j]
            # Dirección 135
            elif (112.5 <= ang[i,j] < 157.5):
                q = magnitud[i-1, j-1]
                r = magnitud[i+1, j+1]

            if (magnitud[i,j] >= q) and (magnitud[i,j] >= r):
                Z[i,j] = magnitud[i,j]
            else:
                Z[i,j] = 0

    # 5. Umbral con histéresis
    fuerte = 255
    debil = 50

    res = np.zeros((M,N), dtype=np.uint8)
    fuerte_i, fuerte_j = np.where(Z >= umbral_alto)
    debil_i, debil_j = np.where((Z <= umbral_alto) & (Z >= umbral_bajo))

    res[fuerte_i, fuerte_j] = fuerte
    res[debil_i, debil_j] = debil

    # Conexión por histéresis
    for i in range(1, M-1):
        for j in range(1, N-1):
            if res[i,j] == debil:
                if ((res[i+1, j-1] == fuerte) or (res[i+1, j] == fuerte) or (res[i+1, j+1] == fuerte)
                    or (res[i, j-1] == fuerte) or (res[i, j+1] == fuerte)
                    or (res[i-1, j-1] == fuerte) or (res[i-1, j] == fuerte) or (res[i-1, j+1] == fuerte)):
                    res[i,j] = fuerte
                else:
                    res[i,j] = 0

    return res

def filtro_laplaciano(imagen: Image.Image) -> np.ndarray:
    """
    Aplica un filtro Laplaciano a la imagen para resaltar bordes.
    
    Parámetros:
    -----------
    imagen : PIL.Image
        Imagen de entrada (RGB o escala de grises).
    
    Retorna:
    --------
    np.ndarray
        Imagen resultante después de aplicar el filtro Laplaciano.
    """
    # Convertir a escala de grises
    gris = imagen.convert("L")
    
    # Kernel Laplaciano clásico (detección de bordes sin dirección)
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # También se puede usar esta versión (más sensible):
    # kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=np.float32)

    # Aplicar convolución
    lap = convolucion(gris, kernel)

    # Normalizar resultado a rango [0,255]
    lap = np.clip(lap, 0, 255).astype(np.uint8)

    return lap