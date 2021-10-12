from easyfacenet.simple.facenet import align_face, embedding
import face_recognition as fcr
from scipy import spatial
import numpy as np
import time
import cv2

class FaceCompare:

    def __init__(self, id_image, selfie_image, threshold=0.7):
        self.id_image = id_image
        self.selfie_image = selfie_image
        self.threshold = threshold

    def facenet(self):
        results = {}
        start_time = time.time()
        images = [self.id_image, self.selfie_image]
        aligned = align_face(images)
        comparisons, scores = self.distance(aligned)
        results = {
            'data':
            {
                'score': round((1 - scores[0][1]),3),
                'threshold': self.threshold,
                'verdict': bool(comparisons[0][1]),
                'elapsedTime (s)': round((time.time() - start_time), 3)
            }
        }
        return results

    def resnet(self):
        results = {}
        start_time = time.time()
        id_face = self.read_image(self.id_image)
        selfie = self.read_image(self.selfie_image)
        id_encoding = fcr.face_encodings(id_face, num_jitters=10)[0]
        face_encoding = fcr.face_encodings(selfie, num_jitters=10)[0]
        verdict = fcr.compare_faces([id_encoding], face_encoding, (1 - self.threshold))
        score = fcr.face_distance([id_encoding], face_encoding)
        results = {
            'data':
            {
                'score': round((1 - score[0]),3),
                'threshold': self.threshold,
                'verdict': bool(verdict[0]),
                'elapsedTime (s)': round((time.time() - start_time), 3)
            }
        }
        return results

    def distance(self, images):
        emb = embedding(images)
        sims = np.zeros((len(images), len(images)))
        scores = np.zeros((len(images), len(images)))

        for i in range(len(images)):
            for j in range(len(images)):
                sims[i][j] = (1 - spatial.distance.cosine(emb[i], emb[j]) > self.threshold)
                scores[i][j] = spatial.distance.cosine(emb[i], emb[j])
        return sims, scores

    def read_image(self, path):
        try:
            return cv2.imread(path)
        except:
            raise Exception("Image not loaded into model")
    