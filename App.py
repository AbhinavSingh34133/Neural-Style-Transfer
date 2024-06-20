from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import Adam_Model  # Import your NST function from Adam_model.py

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    content_image = request.files['content_image']
    style_image = request.files['style_image']

    if content_image.filename == '' or style_image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    content_filename = secure_filename(content_image.filename)
    style_filename = secure_filename(style_image.filename)

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

    content_image.save(content_path)
    style_image.save(style_path)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{content_filename}')
    Adam_Model.perform_nst(content_path, style_path, output_path)  # Assuming you have a function to perform NST

    return jsonify({'output_image': f'/uploads/{os.path.basename(output_path)}'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(debug=True)