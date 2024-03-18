# LeafAI-Django

LeafAI-Django is an AI-powered leaf disease detection platform built with Django. It helps farmers diagnose and manage plant health issues using convolutional neural networks (CNN). Simply upload leaf images to get instant disease severity predictions. A user-friendly interface makes it easy to use, providing accurate results for efficient farming practices.

<img width="960" alt="Screenshot 2024-03-19 000729" src="https://github.com/FouadHellal/Django-LeafDoctor-AI/assets/113594352/9a78f3f1-6f09-42d9-84f5-175f4076f007">
<img width="960" alt="Screenshot 2024-03-19 000550" src="https://github.com/FouadHellal/Django-LeafDoctor-AI/assets/113594352/9ba25546-bc37-4b3c-9cc1-3330e226204a">

## Features

- **AI-Powered Detection**: Utilizes state-of-the-art convolutional neural networks to detect and diagnose leaf diseases accurately.
- **User-Friendly Interface**: Simple and intuitive web interface for easy uploading and viewing of leaf images.
- **Instant Results**: Get instant disease severity predictions with detailed information about the detected issues.
- **Efficient Farming**: Helps farmers make informed decisions for managing plant health, leading to efficient farming practices.

## Installation

Follow these steps to set up LeafAI-Django locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/LeafAI-Django.git
    ```

2. Navigate to the project directory:
    ```bash
    cd LeafAI-Django
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run LeafAI-Django locally, follow these steps:

1. Navigate to the project directory if you haven't already.
2. Run the Django development server:
    ```bash
    python manage.py runserver
    ```
3. Access the application by visiting `http://localhost:8000` in your web browser.

## Understanding views.py

The `views.py` file in our Django project plays a crucial role in handling HTTP requests, processing data, and generating responses. Let's dive into the details of what each part of the code does:

### Uploading Images

When a user uploads an image through our website, the `upload_image` function in `views.py` is triggered. Here's what happens step by step:

1. **Form Validation**: The function checks if the request method is POST and if the form data is valid.

3. **Background Removal**: The uploaded image undergoes background removal using our custom `remove_background` function. This function utilizes computer vision techniques to isolate the leaf from the background, it uses the remove function from *rembg* package.

4. **Image Processing**: The processed image is then converted to the YCbCr color space, and then the Cr plan is taken. After that a clustering algorithm (FCM) is applied to segment the image into two classes, 1 and 0.

5. **Disease Severity Prediction**: Based on the segmented image, our trained AI model predicts the severity of the leaf disease. The model, implemented in the `predict_disease_severity` function, utilizes convolutional neural networks (CNNs) trained on the PlantVillage dataset.

6. **Database Interaction**: Information about the uploaded image and its disease severity is stored in the database using Django's ORM.

7. **Rendering the Result**: Finally, the function renders a response containing the processed image, the percentage of pixels in the sick zone, and the predicted disease severity.

### About the AI Model

Our AI model, stored in the `modelNEW.pkl` file, is trained to classify leaf images into three categories: healthy, sick, and dying. Here are the key components of the model:

- **Architecture**: The model is based on the MobileNetV2 architecture, optimized for efficiency and accuracy.
  
- **Training Data**: We trained the model using the PlantVillage dataset, which contains thousands of labeled images of plant diseases.

- **Prediction**: Given an input image, the model predicts the probability of each class using softmax activation, and the class with the highest probability is selected as the final prediction.

Feel free to explore the `views.py` file for more details on how our Django application processes image uploads and makes predictions about leaf health.

## Contributing

We welcome contributions from the community. If you'd like to contribute to LeafAI-Django, please fork the repository, make your changes, and submit a pull request !

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


