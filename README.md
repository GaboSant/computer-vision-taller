# 📖 Explicación del script de main

Este archivo define los **parámetros de prueba** y permite ejecutar diferentes funciones de verificación implementadas en los módulos `test_color`, `test_camera` y `test_filters`.
De esta forma, podemos comprobar fácilmente cómo se comportan las transformaciones de color, las distorsiones de cámara y los filtros clásicos de visión por computador.

---

## 📌 Importaciones

```python
import numpy as np
from PIL import Image
from tests import test_color, test_camera, test_filters
```

* **numpy (`np`)** → para trabajar con matrices y kernels.
* **PIL.Image** → para manejar imágenes en formato PIL.
* **tests** → contiene los módulos de prueba:

  * `test_color`: pruebas de conversión de color, histogramas y cuantización.
  * `test_camera`: pruebas de distorsión de cámara (radial y focal).
  * `test_filters`: pruebas de convolución, Sobel, Canny y Laplaciano.

---

## 🎨 Parámetros para pruebas de color

```python
sc = "yuv"      # espacio de color: 'hsv', 'lab', 'yuv'
n = 2           # número de imagen: 1 o 2
plot = False    # mostrar imágenes (True) o imprimir matrices (False)
k = [4, 8]      # valores de k para cuantización de color
```

* `sc`: espacio de color al cual se convertirá la imagen (`hsv`, `lab`, `yuv`).
* `n`: elige la imagen de prueba (1 o 2).
* `plot`:

  * `True` → muestra la imagen convertida con matplotlib.
  * `False` → imprime los valores de los canales como matrices NumPy.
* `k`: lista con los valores de **cuantización de colores** (ejemplo: 4 y 8 colores).

Pruebas disponibles:

* `test_color.test_rgb(n, plot)` → muestra los canales RGB.
* `test_color.test_rgb_to_any(sc, n, plot)` → convierte de RGB a otro espacio (`hsv`, `lab`, `yuv`).
* `test_color.test_histogram(n)` → genera histogramas de color.
* `test_color.test_cuantizacion(n, k)` → aplica cuantización de colores.
* `test_color.test_cuantizacion_con_tamano(n, k)` → además muestra el **tamaño en KB** de las imágenes cuantizadas.

---

## 📷 Parámetros para pruebas de cámara

```python
k1 = 0.5       # coeficiente de distorsión radial k1
k2 = -0.5      # coeficiente de distorsión radial k2
f = 0.1        # distancia focal
```

* `k1`, `k2`: controlan la **distorsión radial** (efecto barril o cojín).
* `f`: define la **distancia focal** de la cámara simulada.

Pruebas disponibles:

* `test_camera.test_radial_distortion(n, k1, k2)` → aplica distorsión radial a la imagen.
* `test_camera.test_focal_distortion(n, f)` → aplica distorsión según la distancia focal.

---

## 🧩 Parámetros para pruebas de filtros

```python
kernel = np.array([[0, 2, 0],
                   [2, 8, 2],
                   [0, 2, 0]], dtype=np.float32)
umbral_bajo = 50
umbral_alto = 150
```

* `kernel`: matriz usada para la **convolución genérica** (puede modificarse para aplicar distintos filtros).
* `umbral_bajo` y `umbral_alto`: umbrales para el detector de bordes **Canny**.

Pruebas disponibles:

* `test_filters.test_convolucion(n, kernel)` → aplica convolución genérica.
* `test_filters.test_sobel(n)` → aplica filtros Sobel en X y Y.
* `test_filters.test_canny(n, umbral_bajo, umbral_alto)` → aplica el detector de Canny.
* `test_filters.test_laplacian(n)` → aplica el filtro Laplaciano, que resalta bordes sin dirección específica.

---

## ▶️ Ejecución de las pruebas

Al final del archivo hay bloques comentados para **activar o desactivar pruebas**.
Por ejemplo:

```python
#test_color.test_rgb(n, plot)           # Mostrar canales RGB
#test_camera.test_radial_distortion(n, k1, k2)   # Probar distorsión radial
#test_filters.test_canny(n, umbral_bajo, umbral_alto)  # Probar detector de Canny
```

Para ejecutar una prueba, solo hay que **descomentar** la línea correspondiente.

---

## ✅ Resumen

Este archivo funciona como un **panel de control** para ejecutar distintos experimentos:

* Conversión de color y reducción de colores.
* Simulación de distorsiones de cámara.
* Filtros de detección de bordes y convoluciones personalizadas.

De esta manera, se pueden comparar visualmente y analizar los resultados de cada técnica de procesamiento de imágenes.

---

¿Quieres que te prepare también un **ejemplo de uso en consola** en el README (tipo `python main.py`) para que quede más práctico para quien lo use?
