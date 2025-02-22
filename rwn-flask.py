from flask import Flask, request, jsonify
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
model_check_leaf = tf.keras.models.load_model('rwn-epc100.keras')
model = tf.keras.models.load_model('rwn-epc100v2.keras')

class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Mosaic Virus', 'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus']
class_names_check_leaf = ['leaf', 'notleaf']

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # logging.info("Received request data: %s", data)  # Log the received data

        image_data = data['image']  # Assuming you send the image under the key 'image'
        logging.info("Received image data of length: %d", len(image_data))  # Log the length of the image data

        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        # logging.info("Decoded image size: %s", image.size)
        # logging.info("Decoded image: %s", image)
        # Preprocess the image as required by your model
        image = image.resize((180, 180))  # Resize to the input shape of your model
        # image_array = np.array(image) / 255.0  # Normalize if required
        # image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension if required
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0) # Create a batch

        # Checking the image if its leaf or not
        prediction_check_leaf = model_check_leaf.predict(image_array)
        logging.info("model: %s", prediction_check_leaf)
        score_check_leaf = tf.nn.softmax(prediction_check_leaf[0])
        predicted_class_index_check_leaf = np.argmax(score_check_leaf)  # Get the index of the class with the highest probability
        predicted_label_check_leaf = class_names_check_leaf[predicted_class_index_check_leaf]

        if predicted_label_check_leaf == 'notleaf':
            logging.info("Predicted label: %s", predicted_label_check_leaf)
            return predicted_label_check_leaf, 200
        else:
            # Make prediction
            prediction = model.predict(image_array)
            logging.info("model: %s", prediction)
            score = tf.nn.softmax(prediction[0])
            predicted_class_index = np.argmax(score)  # Get the index of the class with the highest probability
            predicted_label = class_names[predicted_class_index]  # Map index to label

            logging.info("Predicted label: %s", predicted_label)  # Log the predicted label
            logging.info("Accuracy: %s", 100 * np.max(score))  # Log the predicted label

            # Return the predicted label as a plain string
            return predicted_label, 200  # Return the label and HTTP status code 200
    except KeyError as e:
        logging.error("Missing key: %s", str(e))  # Log the error
        return f'Missing key: {str(e)}', 400  # Return error message as string
    except Exception as e:
        logging.error("Error occurred: %s", str(e))  # Log the error
        return str(e), 500  # Return error message as string
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)