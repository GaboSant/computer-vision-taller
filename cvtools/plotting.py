import numpy as np
import matplotlib.pyplot as plt


def plot(canales: list, colores: list, titulos: list) -> None:
    """
    Muestra múltiples imágenes en una cuadrícula de subplots.

    Parámetros:
    -----------
    canales : list of np.ndarray
        Lista de arrays 2D representando las imágenes a mostrar.
    colores : list of str
        Lista de mapas de color para cada imagen.
    titulos : list of str
        Lista de títulos para cada subplot.
    """

    plt.figure(figsize=(15, 5))

    for i, (canal, cmap, titulo) in enumerate(zip(canales, colores, titulos), 1):
        plt.subplot(1, 3, i)
        plt.imshow(canal, cmap=cmap)
        plt.title(titulo)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def canales_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> None:
    # Lista de canales y sus respectivos mapas de color
    canales = [r, g, b]
    colores = ['Reds', 'Greens', 'Blues']
    titulos = ['Canal R (Rojo)', 'Canal G (Verde)', 'Canal B (Azul)']

    # Mostrar los canales en subplots
    plot(canales, colores, titulos)


def canales_hsv(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> None:
    # Lista de canales y sus respectivos mapas de color
    canales = [h, s, v]
    colores = ['hsv', 'grey', 'grey']
    titulos = ['Canal H (Tono)', 'Canal S (Saturación)', 'Canal V (Valor)']

    # Mostrar los canales en subplots
    plot(canales, colores, titulos)


def canales_yuv(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    # Lista de canales y sus respectivos mapas de color
    canales = [y, u, v]
    colores = ['gray', 'winter_r', 'autumn_r']
    titulos = ['Canal Y (Luminancia)', 'Canal U (Crominancia)', 'Canal V (Crominancia)']

    # Mostrar los canales en subplots
    plot(canales, colores, titulos)


def canales_lab(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    # Lista de canales y sus respectivos mapas de color
    canales = [l, a, b]
    colores = ['gray', 'RdYlGn_r', 'PuOr']
    titulos = ['Canal L (Luminosidad)', 'Canal a (Verde-Rojo)', 'Canal b (Azul-Amarillo)']

    # Mostrar los canales en subplots
    plot(canales, colores, titulos)


def mostrar_canales(canales: list, cmap: str) -> None:
    """
    Muestra múltiples canales de una imagen en una cuadrícula de subplots.

    Parámetros:
    -----------
    canales : list of np.ndarray
        Lista de arrays 2D representando los canales a mostrar.
    cmap : str
        Mapa de color a usar para la visualización.
    """
    cmap = cmap.lower()
    match cmap:
        case 'rgb':
            canales_rgb(*canales)
        case 'hsv':
            canales_hsv(*canales)
        case 'yuv':
            canales_yuv(*canales)
        case 'lab':
            canales_lab(*canales)
    