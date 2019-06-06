"""Flask web server for serving lie_detector predictions"""

from flask import Flask, request, jsonify
from flask_cors import CORS
# from tensorflow.keras import backend

ALLOWED_EXTENSIONS = set(['.mp4'])

app = Flask(__name__)
CORS(app)


# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
# with backend.get_session().graph.as_default() as _:
#     predictor = LinePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

# Sanity check.
@app.route('/')
def index():
    return 'Flask server up and running!'

@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    video = _load_video()
    print('success')
    return 'success'
    


def _allowed_file(fname):
    return '.' in fname and fname.split('.')[1].lower() in ALLOWED_EXTENSIONS


def _load_video():
    if request.method == 'POST':
        data = request.files['file']
        if data is not None:
            return 'no file received'
        else:       
            raise ValueError('didnt work')


def main():
    app.run(port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()