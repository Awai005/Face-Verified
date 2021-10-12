import os

from flask import Flask, request
from face_compare import FaceCompare
from werkzeug.utils import secure_filename
from flask import jsonify
import uuid
import PIL.Image as Image
import json
import io
import base64
from http import HTTPStatus
import time
app = Flask(__name__, instance_relative_config=True)

app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)
app.debug = True
# a simple page that says hello
UPLOAD_FOLDER = 'data/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": "Your request could not be processed. Please upload appropriate images",
    })
    response.content_type = "application/json"
    return response

@app.route('/')
def welcome():
    return jsonify ({
        "code": HTTPStatus.OK,
        "message": "face verification online"
    })


def remove_images(images):
    for img in images:
        os.remove(img)
    return images
def params_in(key, params):
    if key not in params:
        return key
    return ''

def save_image(image):
    filename = str(uuid.uuid4()) + '.jpg'
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def save_file(file):
     if file and allowed_file(file.filename):
         filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename).split('.')[-1]
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         return os.path.join(app.config['UPLOAD_FOLDER'], filename)
         
@app.route('/api/facenet', methods=['GET' ,'POST'])
def facenet():
    if request.method == 'GET':
        return jsonify ({
            "code" : HTTPStatus.OK,
            "message": "Use post for this route."
        })
    if request.method == 'POST':
        message = []
        if 'id_image'  or 'selfie_image' not in request.files:
            if 'id_image' not in request.files:
                message.append('id_image is required')
            if 'selfie_image' not in request.files:
                message.append('selfie_image is required')

            if len(request.get_data()) > 0:
                message = []
                content =  json.loads(request.get_data())
                if 'id_image' not in content:
                    message.append('id_image is required')
                if 'selfie_image' not in content: 
                    message.append('selfie_image is required')

        if len(message) > 0:
            return jsonify({
                "errors": message,
                "code": HTTPStatus.UNPROCESSABLE_ENTITY
            })

        if len(request.get_data()) > 0:
            threshold = 0.7
            content =  json.loads(request.get_data())
            id_image = Image.open(io.BytesIO(base64.b64decode(str(content['id_image']))))
            selfie_image = Image.open(io.BytesIO(base64.b64decode(str(content['selfie_image']))))
            if 'threshold' in content:
                threshold = float(content['threshold'])

            id_image_path = save_image(id_image)
            selfie_image_path = save_image(selfie_image)
            face_matcher = FaceCompare(id_image_path, selfie_image_path, threshold)
            results = face_matcher.facenet()
            if results:
                remove_images([id_image_path, selfie_image_path])
            return results

        threshold = request.form.get('threshold', 0.7, type=float)
        id_image_path = save_file (request.files['id_image'])
        selfie_image_path = save_file (request.files['selfie_image'])
        face_matcher = FaceCompare(id_image_path, selfie_image_path, threshold)
        results = face_matcher.facenet()
        if results:
            remove_images([id_image_path, selfie_image_path])
        return results


@app.route('/api/resnet', methods=['GET' ,'POST'])
def resnet():
    if request.method == 'GET':
        return jsonify ({
            "code" : HTTPStatus.OK,
            "message": "Use post for this route."
        })
    if request.method == 'POST':
        message = []
        if 'id_image'  or 'selfie_image' not in request.files:
            if 'id_image' not in request.files:
                message.append('id_image is required')
            if 'selfie_image' not in request.files:
                message.append('selfie_image is required')

            if len(request.get_data()) > 0:
                message = []
                content =  json.loads(request.get_data())
                if 'id_image' not in content:
                    message.append('id_image is required')
                if 'selfie_image' not in content: 
                    message.append('selfie_image is required')

        if len(message) > 0:
            return jsonify({
                "errors": message,
                "code": HTTPStatus.UNPROCESSABLE_ENTITY
            })

        if len(request.get_data()) > 0:
            threshold = 0.7
            content =  json.loads(request.get_data())
            id_image = Image.open(io.BytesIO(base64.b64decode(str(content['id_image']))))
            selfie_image = Image.open(io.BytesIO(base64.b64decode(str(content['selfie_image']))))
            if 'threshold' in content:
                threshold = float(content['threshold'])

            id_image_path = save_image(id_image)
            selfie_image_path = save_image(selfie_image)
            face_matcher = FaceCompare(id_image_path, selfie_image_path, threshold)
            results = face_matcher.resnet()
            if results:
                remove_images([id_image_path, selfie_image_path])
            return results

        threshold = request.form.get('threshold', 0.7, type=float)
        id_image_path = save_file (request.files['id_image'])
        selfie_image_path = save_file (request.files['selfie_image'])
        face_matcher = FaceCompare(id_image_path, selfie_image_path, threshold)
        results = face_matcher.resnet()
        if results:
            remove_images([id_image_path, selfie_image_path])
        return results