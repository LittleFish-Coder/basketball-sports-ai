import time
from threading import Thread
from multiprocessing import Process
import cv2, imutils
from PyQt5.QtCore import QTimer, Qt
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
import multiprocessing
from multiprocessing import shared_memory
from NDItools import cvndi
import torch
from PredictFunction import *
import os

class PTZNDIGet:
    def __init__(self, src, qlabel, width, height):
        self.stream = cvndi.VideoCapture(src)
        self.qlabel = qlabel
        self.width = width
        self.save_path = os.getcwd() + '/save_video/'
        self.height = height
        self.video_writer = None
        # -----------------video info-----------------
        self.grabbed = False
        while not self.grabbed:
            self.grabbed, self.frame = self.stream.read()
        # -----------------video info-----------------
        self.frame_width, self.frame_height = self.frame.shape[1], self.frame.shape[0]
        self.fps = 30
        # -----------------video info-----------------
        self.model = I3DModel()

        # -----------------video info-----------------
        self.stopped = False
        self.kill = False
        self.record = False
        self.re = False

    def start(self):
        self.get_frame = Thread(target=self.get, args=())
        self.get_frame.start()
        return self

    def get(self):
        while True:
            start = time.time()
            self.grabbed, frame = self.stream.read()
            if self.kill:
                print('kill thread')
                break
            if self.stopped or not self.grabbed:
                continue
            # frame = cv2.resize(frame, (1280, 720))
            # if self.record and self.video_writer == None:
            #     self.video_writer = cv2.VideoWriter(self.save_path + '/' + str(len(os.listdir(self.save_path))) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25,
            #                     (self.frame_width, self.frame_height))
            # elif self.record and self.video_writer != None:
            #     self.video_writer.write(frame)
            #     if self.re:
            #         self.video_writer.release()
            #         self.re = False
            #         self.record = False
            #         self.video_writer = None

            # -----------------model info-----------------
            video_frame = self.model.predict(frame[..., ::-1])
            image = qimage2ndarray.array2qimage(video_frame)
            # -----------------video info-----------------
            self.qlabel.setPixmap(QtGui.QPixmap.fromImage(image))



    def kill_thread(self):
        self.kill = True
        self.stream.release()


