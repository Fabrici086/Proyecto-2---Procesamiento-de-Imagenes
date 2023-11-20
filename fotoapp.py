
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2

def solicitar_url_imagen():
  """
  Esta función solicita al usuario que introduzca la URL de una imagen en GitHub.

  Argumento: No recibe parámetros.

  Valor de retorno: Devuelve la URL de la imagen en GitHub.
  """
  url_imagen = input("Por favor, introduce la URL de la imagen en GitHub: ")
  return url_imagen

def descargar_imagen_desde_github(url_imagen):
    """
    Descarga una imagen desde una URL de GitHub y la carga como un objeto de imagen utilizando las bibliotecas requests y Pillow.
    
    Argumento: url_imagen: URL de la imagen en GitHub que se desea descargar (cadena de texto).
    
    Valor de retorno:
        Devuelve la imagen descargada desde la URL de GitHub como un objeto de imagen de la biblioteca Pillow.
    """
    response = requests.get(url_imagen)
    imagen = Image.open(BytesIO(response.content))
    return imagen


def redimensionar_imagen_url(url_imagen, plataforma):
    """
    La función redimensionar_imagen_url permite redimensionar una imagen descargada desde una URL (preferentemente Github) para adaptarla a diferentes plataformas sociales, como Youtube, Instagram, Twitter o Facebook.

    Argumentos:
        url_imagen (str): La URL de la imagen que se va a redimensionar.
        plataforma (str): La plataforma social para la que se va a redimensionar la imagen. Las opciones válidas son: "Youtube", "Instagram", "Twitter" o "Facebook".

    Esta función descarga la imagen desde la URL proporcionada, la redimensiona de acuerdo con las dimensiones predefinidas y guarda la imagen redimensionada con un nombre específico.
    Por último, muestra la imágene original y la redimensionada.
    """
    dimensiones = {
        "Youtube": (1280, 720),
        "Instagram": (1080, 1080),
        "Twitter": (1024, 512),
        "Facebook": (1200, 630)
    }

    response = requests.get(url_imagen)
    imagen = Image.open(BytesIO(response.content))

    if plataforma in dimensiones:
        nueva_dimension = dimensiones[plataforma]

        imagen_redimensionada = imagen.resize(nueva_dimension)

        nombre_redimensionada = f"redimensionada_{plataforma}.jpg"
        imagen_redimensionada.save(nombre_redimensionada)

        print(f"Imagen redimensionada para {plataforma}. Guardada como '{nombre_redimensionada}' con dimensiones: {nueva_dimension}")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(imagen)
        plt.title('Imagen Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imagen_redimensionada)
        plt.title(f'Imagen Redimensionada para {plataforma}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Plataforma no compatible. Las opciones son: Youtube, Instagram, Twitter o Facebook.")

def ajustar_contraste_histograma_color_url(url_imagen):
    """
    Ajusta el contraste y ecualiza el histograma de color de una imagen desde una URL de GitHub.

    Argumentos:
    url_imagen (str): URL de la imagen de GitHub que se procesará.

    Esta función descarga la imagen desde la URL proporcionada, ajusta su contraste y ecualiza su histograma de color.
    Por último, muestra la imagen original y la imagen ecualizada en un gráfico.
    """
    response = requests.get(url_imagen)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    imagen = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    canales = cv2.split(imagen)
    canales_ecualizada = [cv2.equalizeHist(canal) for canal in canales]
    imagen_ecualizada = cv2.merge(canales_ecualizada)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imagen_ecualizada, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Ecualizada')
    plt.axis('off')

    plt.show()

def aplicar_filtro(imagen, nombre_filtro):
    """
    Aplica un filtro específico a una imagen y muestra el resultado.

    Argumentos:
    imagen: La imagen a la que se le aplicará el filtro.
    nombre_filtro (str): El nombre del filtro a aplicar.

    Filtros disponibles:
    1. BLUR: Suaviza la imagen.
    2. CONTOUR: Encuentra los bordes en la imagen.
    3. DETAIL: Realza los detalles de la imagen.
    4. EDGE_ENHANCE: Realza los bordes de la imagen.
    5. EMBOSS: Hace que la imagen parezca grabada.
    6. SHARPEN: Aumenta la nitidez de la imagen.
    7. SMOOTH: Hace que la imagen se vea suave.

    Esta función aplica el filtro solicitado a la imagen y muestra el resultado utilizando matplotlib.
    Por último, guarda la imagen filtrada con un nombre específico.
    """
    filtro = getattr(ImageFilter, nombre_filtro)
    imagen_filtrada = imagen.filter(filtro)
    plt.imshow(imagen_filtrada)
    plt.title(nombre_filtro, color='red')
    plt.axis('off')
    plt.show()
    imagen_filtrada.save(f"imagen_filtrada_{nombre_filtro}.jpg")


def aplicar_todos_filtros(imagen):
    """
    Aplica todos los filtros predefinidos a una imagen y muestra los resultados.

    Argumento:
    imagen: La imagen a la que se le aplicarán los filtros.

    Esta función itera sobre una lista de filtros predefinidos y aplica cada uno de ellos a la imagen.
    Por último, muestra las imágenes filtradas utilizando matplotlib.
    """
    filtros = [
        'BLUR',
        'CONTOUR',
        'DETAIL',
        'EDGE_ENHANCE',
        'EDGE_ENHANCE_MORE',
        'EMBOSS',
        'FIND_EDGES',
        'SHARPEN',
        'SMOOTH'
    ]

    for filtro in filtros:
        aplicar_filtro(imagen, filtro)

def binarizar_imagen(imagen, umbral):
    """
    Binariza una imagen dada utilizando un umbral específico.

    Argumentos:
    imagen: La imagen a binarizar.
    umbral (int): El valor de umbral para la binarización.

    Esta función convierte la imagen a escala de grises, aplica un umbral y crea una nueva imagen binarizada a partir del umbral.
    """
    imagen_gris = imagen.convert('L')
    imagen_array = np.array(imagen_gris)
    imagen_binarizada = np.where(imagen_array > umbral, 255, 0)
    nueva_imagen = Image.fromarray(imagen_binarizada.astype('uint8'))
    edges = nueva_imagen.filter(ImageFilter.FIND_EDGES)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Filtro "Find Edges" en la Imagen Binarizada')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return nueva_imagen
