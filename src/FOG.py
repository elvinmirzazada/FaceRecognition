import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
from src.database.MongoDatabase import Database


class FaceRecognition:

    IMAGES_PATH = 'images'  # put your reference images in here
    CAMERA_DEVICE_ID = 0
    MAX_DISTANCE = 0.51     # increase to make recognition less strict, decrease to make more strict

    def get_face_embeddings_from_image(self, image, convert_to_rgb=False):
        """
        Take a raw image and run both the face detection and face embedding model on it
        """
        # Convert from BGR to RGB if needed
        if convert_to_rgb:
            image = image[:, :, ::-1]

        # run the face detection model to find face locations
        face_locations = face_recognition.face_locations(image)

        # run the embedding model to get face embeddings for the supplied locations
        face_encodings = face_recognition.face_encodings(image, face_locations)

        return face_locations, face_encodings


    def setup_database(self, IMAGES_PATH=None):
        """
        Load reference images and create a database of their face encodings
        """
        database = {}

        for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
            # load image
            image_rgb = face_recognition.load_image_file(filename)

            # use the name in the filename as the identity key
            identity = os.path.splitext(os.path.basename(filename))[0]

            # get the face encoding and link it to the identity
            locations, encodings = self.get_face_embeddings_from_image(image_rgb)
            database[identity] = encodings[0]

        return database


    def paint_detected_face_on_image(self, frame, location, name=None):
        """
        Paint a rectangle around the face and write the name
        """
        # unpack the coordinates from the location tuple
        top, right, bottom, left = location

        if name is None:
            name = 'Unknown'
            color = (0, 0, 255)  # red for unrecognized face
        else:
            color = (0, 128, 0)  # dark green for recognized face

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


    def make_720p(self, video_capture):
        video_capture.set(3, 1280)
        video_capture.set(4, 720)


    # def userFaceMatch(self, known_image, image, isUserExists):
    #

    def getUserFromDatabase(self, fin_code):
        database = Database()
        return database.getUser(fin_code=fin_code)


    def findTheFaces(self, image, fin_code, treshold = 0.49):
        self.MAX_DISTANCE = 1 - treshold
        users = self.getUserFromDatabase(fin_code)
        if users.count() == 0:
            print('Call the IAMAS services')
        else:
            for user in users:
                user_image = cv2.imread(user['image'],1)
                final_image = self.recognize_face(user_image, image, user['fullname'])
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600,600)
                cv2.imshow('image',final_image)
                k = cv2.waitKey(0)
                if k == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                elif k == ord('s'): # wait for 's' key to save and exit
                    cv2.imwrite('image',final_image)
                    cv2.destroyAllWindows()

    def recognize_face(self, known_image, image, name):
        face_locations, face_encodings = self.get_face_embeddings_from_image(image, convert_to_rgb=True)
        known_face_locations, known_face_encodings = self.get_face_embeddings_from_image(known_image, convert_to_rgb=True)
        for location, face_encoding in zip(face_locations, face_encodings):

                # get the distances from this encoding to those of all reference images
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # select the closest match (smallest distance) if it's below the threshold value
                if np.any(distances <= self.MAX_DISTANCE):
                    best_match_idx = np.argmin(distances)
                    name = name
                else:
                    name = None

                # put recognition info on the image
                self.paint_detected_face_on_image(image, location, name)

            # Display the resulting image
        return image


    def run_face_recognition(self, database, CAMERA_DEVICE_ID=0):
        """
        Start the face recognition via the webcam
        """
        # Open a handler for the camera
        video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)

        # the face_recognitino library uses keys and values of your database separately
        known_face_encodings = list(database.values())
        known_face_names = list(database.keys())

        self.make_720p(video_capture)

        ok, frame = video_capture.read()
        cv2.imshow('Video', frame)
        while video_capture.isOpened():
            # Grab a single frame of video (and check if it went ok)
            ok, frame = video_capture.read()
            if not ok:
                logging.error("Could not read frame from camera. Stopping video capture.")
                break

            # run detection and embedding models
            face_locations, face_encodings = self.get_face_embeddings_from_image(frame, convert_to_rgb=True)

            # Loop through each face in this frame of video and see if there's a match
            for location, face_encoding in zip(face_locations, face_encodings):

                # get the distances from this encoding to those of all reference images
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # select the closest match (smallest distance) if it's below the threshold value
                if np.any(distances <= self.MAX_DISTANCE):
                    best_match_idx = np.argmin(distances)
                    name = known_face_names[best_match_idx]
                else:
                    name = None

                # put recognition info on the image
                self.paint_detected_face_on_image(frame, location, name)

            # Display the resulting image
            cv2.imshow('Video', frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
