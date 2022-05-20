import os

from flask import flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from app import app
from detect_face import detect_image_face

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #  print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        rez_detect = detect_image_face(filename)
        return render_template('upload.html', filename=f'static/history/{rez_detect}')
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/history/<filename>')
def display_image(filename):
    #  print('display_image filename: ' + filename)
    return redirect(url_for('static/', filename='history' + filename), code=301)


@app.route('/galary', methods=['GET', 'POST'])
def galary():
    images = list(map(lambda x: f"static/history/{x}", os.listdir('static/history')))
    return render_template('galary.html', images=images)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
