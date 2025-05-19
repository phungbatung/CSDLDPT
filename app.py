from flask import Flask, jsonify, render_template, request, send_from_directory
from search import search_similar_audio
import os
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_audio():
    file = request.files.get('file')
    
    if file is None or file.filename == '':
        return jsonify({"error": "No file provided"}), 400

    input_dir = 'input'
    os.makedirs(input_dir, exist_ok=True)
    save_path = os.path.join(input_dir, file.filename)
    file.save(save_path)
    
    similar_files = search_similar_audio(save_path, top_k=3)
    print(similar_files)
    os.remove(save_path)
    
    # Trả về list URL file có thể nghe trực tiếp
    file_urls = ["http://127.0.0.1:5000/file/" + f[0] for f in similar_files]


    return jsonify({
        "similar_files": file_urls
    }), 200

File_Dir = 'data/dataset'
@app.route('/file/<filename>', methods=['GET'])
def get_file(filename):
    try:
        return send_from_directory(File_Dir, filename, as_attachment=False)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404