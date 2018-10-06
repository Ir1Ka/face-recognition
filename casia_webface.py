#!/usr/bin/env python3
# coding: utf-8

'''
Batch generator for CASIA-WebFace.
'''

__author__ = 'IriKa'

import numpy as np
from PIL import Image
import os
import zipfile
import random
from io import BytesIO

CASI_WEBFACE_PATH = './dataset/CASIA-WebFace.zip'
WIDTH = 50
HEIGHT = 50

class casia_webface:
    '''
    The CASIA-WebFace batch generator.
    '''
    def __init__(self, faces_path=None):
        if faces_path is None:
            faces_path = CASI_WEBFACE_PATH
        self.faces_path = faces_path
        self.faces = zipfile.ZipFile(self.faces_path)
        self.face_names = list(filter(lambda x: not x.endswith('/'), self.faces.namelist()))
        random.shuffle(self.face_names)
        self.index = 0
        self.size = len(self.face_names)
    
    def __enter__(self):
        print('casia_webface enter')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('casia_webface exit')
        self.close()

    def close(self):
        '''
        Close related resources.
        '''
        self.faces.close()

    def image_preprocessing(self, img):
        '''
        The face preprocessing.
        The best way is processing it using tensorflow.
        '''
        if img.mode is not 'RGB':
            print('Warning: The image mode is not RGB, but we need a RGB image, so we need convert it to RGB.')
            img = img.convert('RGB')
        return img

    def tensor_preprocessing(self, tensor):
        '''
        The tensor (numpy.ndarray) preprocessing.
        The best way is processing it using tensorflow.
        '''
        return tensor

    def read_faces(self, face_names):
        '''
        Read face data from the zip file base on the `face_names`.
        '''
        _faces = []
        for face_name in face_names:
            img = self.image_preprocessing(Image.open(BytesIO(self.faces.read(face_name))))
            face = self.tensor_preprocessing(np.array(img))
            face.resize((1,)+face.shape)
            _faces.append(face)
        return np.concatenate(_faces, axis=0)

    def next_batch(self, batch_size=100):
        '''
        Get the next batch.
        '''
        if self.index + batch_size > self.size:
            self.face_names = random.shuffle(self.face_names)
            self.index = 0
        if self.index + batch_size > self.size:
            raise Exception('The number of faces is too small. size =', self.size, 'batch_size=', batch_size)
        face_names = self.face_names[self.index: self.index+batch_size]
        self.index += batch_size
        return self.read_faces(face_names), [os.path.dirname(name) for name in face_names]

def main():
    with casia_webface() as face_data:
        for i in range(10):
            batch, person_names = face_data.next_batch(10)
            print(type(batch), 'shape', batch.shape)
            #print(person_names)
            #print(batch)

if __name__ == '__main__':
    main()
