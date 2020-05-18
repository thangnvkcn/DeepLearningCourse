import cv2
import dlib
import time

image = cv2.imread('face_reg.jpg')

hog_face_detection = dlib.get_frontal_face_detector()
# cnn_face_detection = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
start_time = time.time()
face_hog = hog_face_detection(image, 1)
end_time = time.time()

print('Hog + SVM execution time: ', str(end_time-start_time) )

for face in face_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (w+x, h+y), (0, 255, 0), 2)

#
# start_time = time.time()
#
# face_cnn = cnn_face_detection(image, 1)
# end_time = time.time()
#
# print('Cnn execution time: ' , str(end_time-start_time))
#
# for face in face_cnn:
#     x = face.rect.left()
#     y = face.rect.top()
#     w = face.rect.right()
#     h = face.rect.bottom()
#     cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)


cv2.imshow("image", image)
cv2.waitKey(0)
