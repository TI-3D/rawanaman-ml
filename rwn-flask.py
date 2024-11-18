from flask import Flask, request, jsonify
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('rwn-epc10v2.keras')

class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Mosaic Virus', 'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus']

@app.route('/predict', methods=['POST'])
def predict():
    # data = request.get_json(force=True)
    # # Preprocess the input data as required by your model
    # prediction = model.predict(data)
    # return jsonify(prediction.tolist())
    try:
        data = request.get_json()
        image_data = data['image']  # Assuming you send the image under the key 'image'

        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        # Preprocess the image as required by your model
        image = image.resize((180, 180))  # Resize to the input shape of your model
        image_array = np.array(image) / 255.0  # Normalize if required
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension if required

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)  # Get the index of the class with the highest probability
        predicted_label = class_names[predicted_class_index]  # Map index to label

        # return jsonify({'prediction': predicted_label}), 200
        return predicted_label, 200
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)