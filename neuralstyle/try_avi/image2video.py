import cv2
fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('output.avi',fourcc,20.0,(480,640))
for i in xrange(1,122):
    img = cv2.imread('images_out/video'+str(i)+'.jpg')
    video.write(img)
video.release()
cv2.destroyAllWindows()
