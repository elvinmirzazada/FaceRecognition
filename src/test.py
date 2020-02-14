from src import FOG
import cv2



#
# database = face_rec.setup_database(IMAGES_PATH='../images/database')
# face_rec.run_face_recognition(database)

img = cv2.imread('../images/test/test16.jpg', 1)
# img = cv2.resize(img, (105,150))
face_rec = FOG.FaceRecognition()
# cv2.imshow('image',img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()


# rec_img = face_rec.recognize_face(unknown_image=img)
# cv2.imshow('image',rec_img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',rec_img)
#     cv2.destroyAllWindows()
face_rec.findTheFaces(img,'kanan')
