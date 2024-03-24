from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadImageForm
from .utils import remove_background, rgb_to_ycbcr
import cv2
import numpy as np
from fcmeans import FCM
from .models import UploadedImage
from django.conf import settings
import os
import json
from tensorflow.keras.models import load_model
import os

# Importez la fonction de prédiction de votre script AI
def predict_disease_severity(image_path):
    # Chargez le modèle
    model_path = os.path.join(os.path.dirname(__file__), 'modelDensenet100.h5')
    model = load_model(model_path)
    
    # Fonction de prétraitement de l'image (à adapter selon votre besoin)
    def preprocess_image(image_path, target_size=(224, 224)):
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        return img

    # Prétraitez l'image
    preprocessed_image = preprocess_image(image_path)
    
    # Faites la prédiction
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))

    # Obtenez la classe prédite
    predicted_class = np.argmax(predictions)
    
    # Mapping des classes prédites aux pourcentages de sévérité des maladies
    if predicted_class == 0:
        severity_level = "Healthy"
    elif predicted_class == 1:
        severity_level = "Sick"
    elif predicted_class == 2:
        severity_level = "Dying"
    return severity_level



def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Enregistrer le fichier téléchargé sur le disque
            uploaded_file = request.FILES['image']
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            # Passer le chemin du fichier enregistré à la fonction remove_background
            
            rgb_no_bg = remove_background(file_path)
            #Conversion RGB
            rgb_image = cv2.cvtColor(rgb_no_bg, cv2.COLOR_BGR2RGB)#OpenCv charge les images en BGR donc une conversion en RGB est nécessaire
            #Conversion YCbCr
            image_ycbcr = rgb_to_ycbcr(rgb_image)
            cr=image_ycbcr[:,:,2]
            # Initialisation du FCM avec les centroids personnalisés
            fcm = FCM(n_clusters=2, distance='minkowski')
            fcm.fit(image_ycbcr[:,:,2].reshape(-1,1))
            fcm_centers = fcm.centers
            fcm_labels = fcm.predict(image_ycbcr[:,:,2].reshape(-1,1))
            fcm_labels = fcm_labels.reshape(image_ycbcr[:,:,2].shape)
                    
            # Calculer la surface de la feuille
            surface = np.sum(rgb_no_bg[:, :, 3] != 0)

            # Créer un masque pour les pixels où l'alpha est différent de zéro
            alpha_mask = (rgb_no_bg[:, :, 3] != 0)

            # Compter le nombre de pixels dans chaque classe en utilisant le masque alpha
            pixels_fcm_label_0 = np.sum((fcm_labels == 0) & alpha_mask)
            pixels_fcm_label_1 = np.sum((fcm_labels == 1) & alpha_mask)

            # Calculer la moyenne des valeurs des pixels dans le plan Cr pour chaque classe, en utilisant le masque
            mean_cr_class_0 = np.mean(cr[(fcm_labels == 0) & alpha_mask])
            mean_cr_class_1 = np.mean(cr[(fcm_labels == 1) & alpha_mask])

            # Déterminer quelle classe représente la région malade en comparant les moyennes
            if mean_cr_class_0 > mean_cr_class_1:
                pixels_zone_malade = pixels_fcm_label_0
              
            else:
                pixels_zone_malade = pixels_fcm_label_1
               
            # Calculer le pourcentage de pixels dans la zone malade par rapport à la surface de la feuille
            pourcentage_pixels_zone_malade = (pixels_zone_malade / surface) * 100
            pourcentage_pixels_zone_malade = "{:.1f}".format(pourcentage_pixels_zone_malade)


            # Sauvegarde des données dans la base de données
            uploaded_image = UploadedImage(image=uploaded_file, percentage=pourcentage_pixels_zone_malade)
            uploaded_image.save()

            image_url = os.path.join(settings.MEDIA_URL, uploaded_file.name)

            severity_level = predict_disease_severity(file_path)

            # Pass severity_level to the template
            return render(request, 'upload_result.html', {'image_url': image_url, 'pourcentage_pixels_zone_malade': pourcentage_pixels_zone_malade, 'severity_level': severity_level})

    else:
        form = UploadImageForm()
    return render(request, 'upload_form.html', {'form': form})
