from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
import joblib
import traceback
import string

app = Flask(__name__, static_url_path='')
CORS(app)

try:
    
    print("Loading model and label encoder...")
    model = load_model('letter_recognition_model.h5')
    label_encoder = joblib.load('label_encoder.joblib')
    print("Model and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model or label encoder: {str(e)}")
    raise

def scale_quadrant(image_array, target_size=(28, 28)):
    """Recursively scale down image by dividing into quadrants"""
    current_height, current_width = image_array.shape
    
    
    if current_height == target_size[0] and current_width == target_size[1]:
        return image_array
        
    
    if current_height < target_size[0] * 2 or current_width < target_size[1] * 2:
        return average_pixels(image_array, target_size)
    
    
    new_height = max(current_height // 2, target_size[0])
    new_width = max(current_width // 2, target_size[1])
    
    
    top_left = image_array[:current_height//2, :current_width//2]
    top_right = image_array[:current_height//2, current_width//2:]
    bottom_left = image_array[current_height//2:, :current_width//2]
    bottom_right = image_array[current_height//2:, current_width//2:]
    
    
    scaled_top_left = scale_quadrant(top_left, target_size)
    scaled_top_right = scale_quadrant(top_right, target_size)
    scaled_bottom_left = scale_quadrant(bottom_left, target_size)
    scaled_bottom_right = scale_quadrant(bottom_right, target_size)
    
    
    top = np.concatenate((scaled_top_left, scaled_top_right), axis=1)
    bottom = np.concatenate((scaled_bottom_left, scaled_bottom_right), axis=1)
    combined = np.concatenate((top, bottom), axis=0)
    
    
    if combined.shape[0] > target_size[0] or combined.shape[1] > target_size[1]:
        return average_pixels(combined, target_size)
    
    return combined

def average_pixels(image_array, target_size):
    """Scale down image by averaging pixel values in each region"""
    current_height, current_width = image_array.shape
    result = np.zeros(target_size)
    
    
    region_height = current_height // target_size[0]
    region_width = current_width // target_size[1]
    
    
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            
            start_h = i * region_height
            end_h = (i + 1) * region_height
            start_w = j * region_width
            end_w = (j + 1) * region_width
            
            
            region = image_array[start_h:end_h, start_w:end_w]
            result[i, j] = np.mean(region)
    
    return result

def process_base64_image(base64_string):
    """Process base64 image data and return scaled 28x28 matrix"""
    
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img = img.convert('L')  
    img_array = np.array(img)
    
    if img_array.shape != (448, 448):
        raise ValueError("Input image must be 448x448 pixels")
    
    
    result = scale_quadrant(img_array, (28, 28))
    return result.astype(np.uint8)


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
            
        image_data = data['image']
        
        
        print("Processing image...")
        result_matrix = process_base64_image(image_data)
        print(f"Matrix shape: {result_matrix.shape}")
        
        
        input_data = result_matrix.reshape(1, 28, 28, 1)
        input_data = input_data / 255.0  
        print("Input data prepared for model")
        
        
        print("Making prediction...")
        prediction = model.predict(input_data)
        predicted_class = int(prediction.argmax())  
        predicted_letter = str(label_encoder.inverse_transform([predicted_class])[0])  
        confidence = float(prediction[0][predicted_class])
        print(f"Predicted letter: {predicted_letter} with confidence: {confidence}")
        
        
        matrix_list = result_matrix.astype(float).tolist()
        
        return jsonify({
            'matrix': matrix_list,
            'prediction': predicted_letter,
            'confidence': confidence
        })
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Failed to process image: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 