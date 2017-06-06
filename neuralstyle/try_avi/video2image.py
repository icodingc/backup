import cv2
from scipy import misc
capture = cv2.VideoCapture("test.avi")
cnt = 0
if capture.isOpened():
    while True:
        ret, prev = capture.read()
        if ret==True:
            cnt +=1
            img = misc.imresize(prev,(640,480))
            print img.shape
            misc.imsave('images_/video'+str(cnt)+'.jpg',img)
        else:break
        if cv2.waitKey(20)==27:break
print cnt
capture.release()
cv2.destroyAllWindows()

