import cv2
import os

cam = cv2.VideoCapture(0)


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press  ==>  ')



count = 0

while(True):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1


        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', cv2.resize(frame,(300,300)))

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break


print("\n 30 нафотал")
cam.release()
cv2.destroyAllWindows()