from PIL import Image
import numpy as np


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    coef = np.array([0.2989, 0.5870, 0.1140])
    grayscale_image = np.dot(image[..., :3], coef)
    grayscale_image = (grayscale_image / grayscale_image.max()) * 255
    return grayscale_image.astype(np.uint8)


def binarize_image(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    binarized_image = np.where(image > threshold, 255, 0)
    return binarized_image.astype(np.uint8)


# Exemplo de uso
# Carregar uma imagem colorida
image_path = 'image.jpg'
color_image = np.array(Image.open(image_path))

# Converter a imagem colorida para tons de cinza
grayscale_image = rgb_to_grayscale(color_image)

# Binarizar a imagem em tons de cinza
binarized_image = binarize_image(grayscale_image)

# Salvar as imagens
grayscale_img_pil = Image.fromarray(grayscale_image)
grayscale_img_pil.save('image_gray.jpg')

binarized_img_pil = Image.fromarray(binarized_image)
binarized_img_pil.save('image_binarized.jpg')
