from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import numpy as np
from keras.models import load_model
import cv2

# Import DatasetHandler from structures module
from structures import DatasetHandler

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None

# Define classes dictionary
classes = {
    0: 'Band Pull Apart',
    1: 'Barbell Dead Row',
    2: 'Barbell Row',
    3: 'Barbell Shrug',
    4: 'Burpees',
    5: 'Clean And Press',
    6: 'Deadlift',
    7: 'Diamond Pushup',
    8: 'Drag Curl',
    9: 'Dumbbell Biceps Curls',
    10: 'Dumbbell Curl Trifecta',
    11: 'Dumbbell Hammer Curls',
    12: 'Dumbbell High Pulls',
    13: 'Dumbbell Overhead Shoulder Press',
    14: 'Dumbbell Reverse Lunge',
    15: 'Dumbbell Scaptions',
    16: 'Man Maker',
    17: 'Mule Kick',
    18: 'Neutral Overhead Shoulder Press',
    19: 'One Arm Row',
    20: 'Overhead Extension Thruster',
    21: 'Overhead Trap Raises',
    22: 'Pushup',
    23: 'Side Lateral Raise',
    24: 'Squat',
    25: 'Standing Ab Twists',
    26: 'W Raise',
    27: 'Walk The Box'
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bgimage.jpg')
def send_bgimage():
    return send_from_directory('templates', 'bgimage.jpg')

@app.route('/results.html')
def results():
    return render_template('results.html')

@app.route('/load_model', methods=['GET'])
def load_model_route():
    global model
    if model is None:
        try:
            model = load_model('model.h5')
            return jsonify( {'status': 'Your model ready to use.'} )
    
        except Exception as e:
            return jsonify({'status': f'Error loading model: {str(e)}'}), 500
        
    else:
        return jsonify( { 'status': f"Model has been already loaded!" } )

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify( { 'success': False } ), 400
    
    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
    video.save(video_path)
    return jsonify( { 'success': True } )

# Define route for getting analysis
@app.route('/get_analysis', methods=['GET'])
def get_analysis():
    global model
    if model is None:
        return jsonify({'error': 'Please load your model to make an detection'}), 500

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')

    dataset_handler = DatasetHandler(False, None, 75, 128, 128, 1, None, None)
    frames = dataset_handler.read_video(video_path)
    frames = np.expand_dims(np.array(frames), axis=0)
    predictions = model.predict(frames)
    max_index = np.argmax(predictions, axis=-1)[0]
    accuracy = predictions[0][max_index] * 100

    analysis_results = {
        'video_path': url_for('uploaded_video', filename='uploaded_video.mp4'),
        'class_name': classes[max_index],
        'accuracy': f'{accuracy:.2f}%'
    }
    return jsonify(analysis_results)

@app.route('/uploaded_videos/<filename>')
def uploaded_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def launch_website():
    app.run(debug=True)
    
    return


def main():
    launch_website()
    
    return 0


if __name__ == '__main__':
    main()
