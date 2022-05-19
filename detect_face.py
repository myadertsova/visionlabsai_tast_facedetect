import datetime

import cv2
import imutils
import numpy
from mtcnn.mtcnn import MTCNN


def upload_image(filename):
    image = cv2.imread('static/uploads/' + filename)
    return image


def detect_image_face(filename):
    detector = MTCNN()
    image = upload_image(filename)
    new_file_name = f'people_detected{datetime.datetime.now()}.png'
    if image.shape[0] < image.shape[1]:
        image = imutils.resize(image, height=1000)
    else:
        image = imutils.resize(image, width=1000)
    image_size = numpy.asarray(image.shape)[0:2]

    faces_boxes = detector.detect_faces(image)

    # Копия изображения для рисования рамок на нём
    image_detected = image.copy()

    # Работа с лицами
    if faces_boxes:

        face_n = 0  # Инициализация счётчика лиц
        for face_box in faces_boxes:

            # Увеличение счётчика файлов
            face_n += 1

            # Координаты лица
            x, y, w, h = face_box['box']

            # Отступы для увеличения рамки
            d = h - w  # Разница между высотой и шириной
            w = w + d  # Делаем изображение квадратным
            x = numpy.maximum(x - round(d / 2), 0)
            x1 = numpy.maximum(x - round(w / 4), 0)
            y1 = numpy.maximum(y - round(h / 4), 0)
            x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
            y2 = numpy.minimum(y + h + round(h / 4), image_size[0])

            # Отборка лиц {selected|rejected}
            if face_box['confidence'] > 0.99:  # 0.99 - уверенность сети в процентах что это лицо

                # Рисует белый квадрат на картинке по координатам
                cv2.rectangle(
                    image_detected,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255, 1),
                    10
                )

            else:

                # Рисует красный квадрат на картинке по координатам
                cv2.rectangle(
                    image_detected,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255, 1),
                    10
                )

        # Сохранение исходного изображения с выделенными лицами
        cv2.imwrite(f'static/history/{new_file_name}', image_detected)
    return new_file_name

