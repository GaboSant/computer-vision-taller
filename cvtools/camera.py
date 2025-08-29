import numpy as np
import cv2

def apply_radial_distortion(image: np.ndarray, k1: float = 0.0, k2: float = 0.0, 
                           interpolation: int = cv2.INTER_LINEAR, 
                           border_mode: int = cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Aplica distorsión radial a una imagen usando el modelo de distorsión de lente.
    
    Args:
        image (np.ndarray): Imagen de entrada como array NumPy
        k1 (float): Primer coeficiente de distorsión radial
        k2 (float): Segundo coeficiente de distorsión radial
        interpolation (int): Método de interpolación
        border_mode (int): Método para manejar bordes
    
    Returns:
        np.ndarray: Imagen con distorsión radial aplicada
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("La imagen debe ser un array NumPy")
    
    if image.size == 0:
        raise ValueError("La imagen de entrada está vacía")
    
    # Obtener dimensiones de la imagen
    height, width = image.shape[:2]
    
    # Crear coordenadas normalizadas del plano de la imagen
    cx, cy = width / 2.0, height / 2.0
    
    # Crear malla de coordenadas
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Coordenadas normalizadas respecto al centro óptico
    xn = (X - cx) / cx
    yn = (Y - cy) / cy
    
    # Calcular distancia radial desde el centro
    r_squared = xn**2 + yn**2
    r = np.sqrt(r_squared)
    
    # Aplicar modelo de distorsión radial
    distortion_factor = 1.0 + k1 * r_squared + k2 * r_squared**2
    
    # Coordenadas distorsionadas normalizadas
    xd_normalized = xn * distortion_factor
    yd_normalized = yn * distortion_factor
    
    # Convertir de vuelta a coordenadas de píxeles
    xd = xd_normalized * cx + cx
    yd = yd_normalized * cy + cy
    
    # Crear mapas de remapeo
    map_x = xd.astype(np.float32)
    map_y = yd.astype(np.float32)
    
    # Aplicar la transformación usando remapeo
    distorted_image = cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode)
    
    return distorted_image

def apply_focal_distortion(image: np.ndarray, new_focal_length: float, 
                          original_focal_length: float = 1.0, 
                          interpolation: int = cv2.INTER_LINEAR, 
                          border_mode: int = cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Aplica distorsión a una imagen simulando un cambio en la distancia focal.
    
    Args:
        image (np.ndarray): Imagen de entrada como array NumPy
        new_focal_length (float): Nueva distancia focal (unidades)
        original_focal_length (float): Distancia focal original
        interpolation (int): Método de interpolación
        border_mode (int): Método para manejar bordes
    
    Returns:
        np.ndarray: Imagen con distorsión por cambio de distancia focal
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("La imagen debe ser un array NumPy")
    
    if image.size == 0:
        raise ValueError("La imagen de entrada está vacía")
    
    if new_focal_length <= 0:
        raise ValueError("La distancia focal debe ser positiva")
    
    # Si la relación es 1, no hay distorsión
    if abs(new_focal_length - original_focal_length) < 1e-6:
        return image.copy()
    
    # Obtener dimensiones de la imagen
    height, width = image.shape[:2]
    
    # Centro de la imagen
    cx, cy = width / 2.0, height / 2.0
    
    # Crear malla de coordenadas
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Coordenadas normalizadas respecto al centro
    xn = (X - cx) / cx
    yn = (Y - cy) / cy
    
    # Calcular distancia radial desde el centro
    r = np.sqrt(xn**2 + yn**2)
    
    # Aplicar transformación basada en la relación focal
    if original_focal_length != 1.0:
        theta = np.arctan2(r, original_focal_length)
    else:
        theta = np.arctan(r)
    
    # Nueva distancia radial basada en la nueva focal
    r_new = new_focal_length * np.tan(theta)
    
    # Factor de escalado radial
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_factor = np.divide(r_new, r, where=r != 0)
        scale_factor[r == 0] = 1.0
    
    # Coordenadas distorsionadas normalizadas
    xd_normalized = xn * scale_factor
    yd_normalized = yn * scale_factor
    
    # Convertir de vuelta a coordenadas de píxeles
    xd = xd_normalized * cx + cx
    yd = yd_normalized * cy + cy
    
    # Crear mapas de remapeo
    map_x = xd.astype(np.float32)
    map_y = yd.astype(np.float32)
    
    # Aplicar la transformación
    distorted_image = cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode)
    
    return distorted_image