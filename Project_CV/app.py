import os
from flask import Flask, render_template, request, send_file
from predictor import check
from segmentation import segmentation

app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
@app.route('/index')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)
    # status = check(filename)
    results = check(filename)
    segmentation(filename)
    return render_template('complete.html', image_name=filename, results=results)


@app.route('/get_image')
def get_image():
    image_path = 'segmentation_results/binary_mask.png'
    return send_file(image_path, mimetype='image/png')
if __name__ == "main":
    app.run(port=4555, debug=True)
