import numpy as np
from PIL import Image
from tests import test_color, test_camera, test_filters

# Parámetros para la prueba de conversión de color
sc = "yuv"      # espacio de color: 'hsv', 'lab', 'yuv'
n = 2           # número de imagen: 1 o 2
plot = False     # mostrar imágenes (True) o imprimir matrices (False)
k = [4, 8]   # valores de k para cuantización de color (usar lista)


# Parámetros para la prueba de camara
k1 = 0.5       # coeficiente de distorsión radial k1
k2 = -0.5       # coeficiente de distorsión radial k2
f = 0.1         # distancia focal

# Parámetros para la prueba de filtros
kernel = np.array([[0, 2, 0],
                    [2, 8, 2],
                    [0, 2, 0]], dtype=np.float32)
umbral_bajo = 50
umbral_alto = 150



# Ejecutar las pruebas
#-------------------------------
# Pruebas de conversión de color (↓ descomentar para ejecutar ↓)
#test_color.test_rgb(n, plot)
#test_color.test_rgb_to_any(sc, n, plot)
#test_color.test_histogram(n)
#test_color.test_cuantizacion(n, k)
#test_color.test_cuantizacion_con_tamano(n, k)

#-------------------------------
# Prueba de distorsión camara (↓ descomentar para ejecutar ↓)
#test_camera.test_radial_distortion(n, k1, k2)
#test_camera.test_focal_distortion(n, f)

#-------------------------------
# Prueba de filtros (↓ descomentar para ejecutar ↓)
#test_filters.test_convolucion(n, kernel)
#test_filters.test_sobel(n)
#test_filters.test_canny(n, umbral_bajo, umbral_alto)
#test_filters.test_laplacian(n)