from flask import Flask, request, jsonify, render_template
from torch_utils import transform_image, get_prediction
import os

app = Flask(__name__, static_url_path = '/static')

@app.route('/')
def render_page():
  return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadajax', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files.get('file')
    # file = request.files['file']

    if file is None or file.filename == '':
      return jsonify('No file found')
  
    if not allowed_file(file.filename):
      return jsonify('File format not supported')

    save_location = os.path.join('images/', file.filename)
    file.save(save_location)

    try:
      with open(save_location, 'rb') as imageFile:
        img_bytes = imageFile.read()
      tensor = transform_image(img_bytes)
      prediction = get_prediction(tensor)
      result = prediction.item()
      return jsonify(["This histopathology image DOESN'T HAVE CANCEROUS cells.", "This histopathology image has CANCEROUS cells."][result])
      # return jsonify(result)
      # return jsonify({'save_location': save_location, 'result': result})

    except:
      return jsonify('Error during prediction')


if __name__ == '__main__':
  app.run(debug=True, port=os.getenv('PORT',5000))