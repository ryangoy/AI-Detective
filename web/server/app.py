"""Flask web server for serving lie_detector predictions"""

from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os
import sys
import time
sys.path.insert(0, './')
sys.path.insert(0, '../')
from lie_detector.predict import predict_example
from tensorflow.keras import backend
import boto3
from botocore.exceptions import ClientError
from decimal import Decimal

app = Flask(__name__)
cors = CORS(app)
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('cydm-actions')

ALLOWED_EXTENSIONS = set(['mp4'])
app.config['UPLOAD_FOLDER'] = '/tmp'

# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
# with backend.get_session().graph.as_default() as _:
#     predictor = LinePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

# Sanity check.
@app.route('/')
def index():
    return 'Flask server up and running!'


@app.route('/get-presigned-post/<filename>', methods=['GET'])
@cross_origin()
def create_presigned_post(filename):
    """Generate a presigned URL S3 POST request to upload a file

    :param bucket_name: string
    :param object_name: string
    :param fields: Dictionary of prefilled form fields
    :param conditions: List of conditions to include in the policy
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Dictionary with the following keys:
        url: URL to post to
        fields: Dictionary of form fields and values to submit with the POST
    :return: None if error.
    """
    params = {
        'Bucket': 'cydm-videos',
        'Key': filename,
        'ContentType': 'mp4',
        'ACL': 'public-read'
    }
    
    expiration=100
    # Generate a presigned S3 POST URL
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('put_object',
                                                    Params=params,
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None

    # The response contains the presigned URL and required fields
    return jsonify({'url':response})


@app.route('/predict/<fname>', methods=['GET'])
@cross_origin()
def face_percent(fname):
    table.put_item(
        Item={
            'id': fname,
            'stage': 'file upload',
            'prediction': None
        }
    )
    download_from_s3(fname)
    vpath = '/tmp/' + fname
    percent = predict_example(vpath, table=table, fname=fname)#, socketio)
    table.update_item(
        Key={
            'id': fname
        },
        UpdateExpression='SET prediction = :val1, stage= :val2',
        ExpressionAttributeValues={
            ':val1': Decimal(percent),
            ':val2': 'finished'
        }
    )
    return jsonify({'status': 'success'})


@app.route('/poll-stage/<fname>', methods=['GET'])
@cross_origin()
def poll_status(fname):
    response = table.get_item(
        Key={
            'id':fname
        })
    item = response['Item']
    pred = None
    if item['prediction']:
        pred = float(item['prediction'])
    return jsonify({'stage': str(item['stage']), 'prediction': pred}) 


def _allowed_file(fname):
    return '.' in fname and fname.split('.')[1].lower() in ALLOWED_EXTENSIONS


def _load_video():
    if 'file' not in request.files:
        print('No file part')
        return None
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return None
    if file and _allowed_file(file.filename):
        filename = file.filename
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fpath)
        upload_to_s3(filename)
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            print("{} has file size {}".format(f, os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))))
        return fpath


def upload_to_s3(fname):
    print('Uploading {} to s3...'.format(fname))
    bucket = 'cydm-videos'
    saved_path = os.path.join('/tmp', fname)
    s3_client = boto3.client('s3')
    s3_client.upload_file(saved_path, bucket, fname)


def download_from_s3(fname):
    print('Downloading {} from s3...'.format(fname))
    bucket = 'cydm-videos'
    save_path = os.path.join('/tmp', fname)
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, fname, save_path)


def main():
    app.run(host='0.0.0.0', port=8000)  # nosec


if __name__ == '__main__':
    main()