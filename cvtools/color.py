import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from cvtools.plotting import mostrar_canales


def rgb_a_hsv(imagen_pil: Image.Image, plot: bool) -> np.ndarray:
    """
    Convierte una imagen de RGB a HSV usando OpenCV.
    """
    rgb_np = np.array(imagen_pil)  # PIL -> NumPy (RGB)
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv_np)

    if plot:
        mostrar_canales([h, s, v], "HSV")
    else:
        print("Canal H:\n", h, "\n")
        print("Canal S:\n", s, "\n")
        print("Canal V:\n", v, "\n")

    return [h, s, v]


def rgb_a_lab(imagen_pil: Image.Image, plot: bool) -> np.ndarray:
    """
    Convierte una imagen de RGB a LAB usando OpenCV.
    """
    rgb_np = np.array(imagen_pil)
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab_np)

    if plot:
        mostrar_canales([l, a, b], "LAB")
    else:
        print("Canal L:\n", l, "\n")
        print("Canal A:\n", a, "\n")
        print("Canal B:\n", b, "\n")

    return [l, a, b]


def rgb_a_yuv(imagen_pil: Image.Image, plot: bool) -> np.ndarray:
    """
    Convierte una imagen de RGB a YUV usando OpenCV.
    """
    rgb_np = np.array(imagen_pil)
    yuv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YUV)

    y, u, v = cv2.split(yuv_np)

    if plot:
        mostrar_canales([y, u, v], "YUV")
    else:
        print("Canal Y:\n", y, "\n")
        print("Canal U:\n", u, "\n")
        print("Canal V:\n", v, "\n")

    return [y, u, v]

def histograma_colores(imagen_pil: Image.Image):
    """
    Calcula y grafica el histograma de colores (RGB) de una imagen.

    Parámetros:
    -----------
    imagen_pil : PIL.Image
        Imagen cargada en formato RGB.
    """
    # Convertir la imagen a arreglo NumPy (RGB)
    imagen_np = np.array(imagen_pil.convert("RGB"))

    # Separar los canales
    r, g, b = imagen_np[:, :, 0], imagen_np[:, :, 1], imagen_np[:, :, 2]

    # Calcular histogramas (256 bins, valores de 0 a 255)
    hist_r, _ = np.histogram(r, bins=256, range=(0, 255))
    hist_g, _ = np.histogram(g, bins=256, range=(0, 255))
    hist_b, _ = np.histogram(b, bins=256, range=(0, 255))

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(hist_r, color='red', label="Rojo")
    plt.plot(hist_g, color='green', label="Verde")
    plt.plot(hist_b, color='blue', label="Azul")

    plt.title("Histograma de colores")
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    return hist_r, hist_g, hist_b

def cuantizacion_simple(imagen_pil: Image.Image, niveles: int) -> Image.Image:
    """
    Aplica cuantización de colores a una imagen, reduciendo el número de colores.

    Parámetros:
    -----------
    imagen_pil : PIL.Image
        Imagen cargada en formato RGB.
    niveles : int
        Número de niveles de cuantización (ej. 256, 64, 16).

    Retorna:
    --------
    PIL.Image
        Imagen cuantizada.
    """
    # Convertir a NumPy
    imagen_np = np.array(imagen_pil.convert("RGB"))

    # Factor de cuantización
    factor = 256 // niveles

    # Cuantizar: dividir → truncar → multiplicar → centrar en nivel
    imagen_cuant = (imagen_np // factor) * factor + factor // 2

    # Convertir de vuelta a imagen PIL
    return Image.fromarray(imagen_cuant.astype(np.uint8), "RGB")

def reducir_peso(imagen_pil: Image.Image, niveles: int, formato: str = "JPEG") -> tuple[Image.Image, float]:
    """
    Reduce el peso de una imagen disminuyendo la cantidad de colores.

    Parámetros:
    -----------
    imagen_pil : PIL.Image
        Imagen original.
    niveles : int
        Número de niveles de cuantización por canal.
    formato : str
        Formato en el que se guarda la imagen temporalmente (ej. "JPEG", "PNG").

    Retorna:
    --------
    (PIL.Image, float)
        Imagen cuantizada y tamaño en KB.
    """
    # Aplicar cuantización
    img_cuant = cuantizacion_simple(imagen_pil, niveles)

    # Guardar en buffer de memoria
    buffer = io.BytesIO()
    img_cuant.save(buffer, format=formato)
    size_kb = len(buffer.getvalue()) / 1024  # tamaño en KB

    return img_cuant, size_kb