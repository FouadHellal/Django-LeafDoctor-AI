
from rembg import remove #utilisé pour la suppression de l'arrière-plan (background)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def remove_background(file_path):
    # Ouvrir l'image à partir du chemin du fichier
    with Image.open(file_path) as image:
        # Convertir l'image en format accepté par la fonction remove
        image_np = np.array(image)
        # Supprimer l'arrière-plan de l'image
        image_no_bg = remove(image_np)
        resized_image = cv2.resize(image_no_bg, (255,255))
    return resized_image

# Conversion RGB --> YCbCr
def rgb_to_ycbcr(rgb_image):
    pixels = np.array(rgb_image) #convertit l'image en un tableau de pixels numpy.
    ycbcr_pixels = np.zeros_like(pixels, dtype=np.uint8) #crée un tableau de la même forme que l'image RGB 
    for i in range(pixels.shape[0]): #Lignes
        for j in range(pixels.shape[1]): #Colonnes
            pixel = pixels[i, j]
            Y = np.clip(0.257 * pixel[0] + 0.504 * pixel[1] + 0.098 * pixel[2] + 16, 0, 255) # np.clip est utilisé pour s'assurer que les valeurs résultantes sont comprises entre 0(valeur min) et 255(valeur max).
            Cb = np.clip(-0.148 * pixel[0] - 0.291 * pixel[1] + 0.439 * pixel[2] + 128, 0, 255)
            Cr = np.clip(0.439 * pixel[0] - 0.368 * pixel[1] - 0.071 * pixel[2] + 128, 0, 255)
            ycbcr_pixels[i, j] = [Y, Cb, Cr] #Les valeurs Y, Cb et Cr calculées sont assignées au pixel correspondant dans l'image de sortie.
    return ycbcr_pixels
