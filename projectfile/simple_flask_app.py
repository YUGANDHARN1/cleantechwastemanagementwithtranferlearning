from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def create_mock_predictions():
    """Create realistic mock predictions for demonstration"""
    import random
    
    categories = ['Recyclable', 'Organic', 'Hazardous', 'Non-Recyclable']
    
    # Generate random but realistic probabilities
    probs = [random.uniform(0.1, 0.9) for _ in range(4)]
    total = sum(probs)
    probs = [p/total for p in probs]  # Normalize to sum to 1
    
    # Sort by probability (highest first)
    sorted_indices = sorted(range(4), key=lambda i: probs[i], reverse=True)
    
    results = []
    for idx in sorted_indices:
        results.append({
            'category': categories[idx],
            'confidence': float(probs[idx]) * 100,
            'probability': float(probs[idx])
        })
    
    return results

def preprocess_image(image):
    """Basic image preprocessing"""
    try:
        # Resize image to standard size
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Basic normalization
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image):
    """Make prediction on image (using mock predictions for now)"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None
        
        # For demonstration, return mock predictions
        # In a real scenario, this would use the trained model
        results = create_mock_predictions()
        
        return results
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Load and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Make prediction
        results = predict_image(image)
        
        if results is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        return jsonify({
            'success': True,
            'predictions': results,
            'filename': file.filename,
            'message': 'Classification completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/portfolio-details')
def portfolio_details():
    """Portfolio details page"""
    return render_template('portfolio-details.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Waste Management Classification System is running',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    
    print("Starting Waste Management Classification System...")
    print("- Flask server initializing...")
    print("- Upload directory created")
    print("- Mock prediction system ready")
    print("- Access the application at http://localhost:5000")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)