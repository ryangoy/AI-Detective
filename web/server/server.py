"""Flask web server for serving lie_detector predictions"""

from flask import Flask, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import sys
sys.path.insert(0, './')
from lie_detector.video_face_detector import generate_cropped_face_video 
# from tensorflow.keras import backend

ALLOWED_EXTENSIONS = set(['mp4'])


app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
# with backend.get_session().graph.as_default() as _:
#     predictor = LinePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

# Sanity check.
@app.route('/')
def index():
    return 'Flask server up and running!'

@app.route('/dev/face_percent', methods=['POST'])
def face_percent():
    vpath = _load_video()
    percent = generate_cropped_face_video(vpath)
    print(percent)
    return jsonify({'percent': percent})


def _allowed_file(fname):
    return '.' in fname and fname.split('.')[1].lower() in ALLOWED_EXTENSIONS


def _load_video():

    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return None
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return None
        if file and _allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fpath)

            return fpath


def main():
    app.run(port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()