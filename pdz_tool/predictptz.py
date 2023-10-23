import time
from threading import Thread
from multiprocessing import Process
import cv2, imutils
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer, Qt
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
import multiprocessing
import json
from multiprocessing import shared_memory
from NDItools import cvndi
import torch
from PredictFunction import *
from combine import *
import os

def find_continuous_blocks_with_ranges(array):
    count = 0
    is_block = False
    blocks = []
    start = -1
    for i, element in enumerate(array):
        if element != 0 and not is_block:
            is_block = True
            count += 1
            start = i
        elif element == 0 and is_block:
            is_block = False
            blocks.append((start, i-1))
    if is_block:  # 處理最後一個連續區塊
        blocks.append((start, len(array)-1))
    return count, blocks


class PTZNDIGet:
    def __init__(self, src, qlabel, diagram, width, height):
        self.stream = cvndi.VideoCapture(src)
        self.qlabel = qlabel
        self.diagram = diagram
        self.width = width
        self.save_path = os.getcwd() + '/save_video/'
        self.height = height
        self.video_writer = False
        # -----------------video info-----------------
        self.grabbed = False
        while not self.grabbed:
            self.grabbed, self.frame = self.stream.read()
        # -----------------video info-----------------
        self.frame_width, self.frame_height = self.frame.shape[1], self.frame.shape[0]
        self.fps = 30
        # -----------------video info-----------------
        self.yolo = ObjectDetection()
        self.arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
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
            self.grabbed, frame = self.stream.read()
            if self.kill:
                print('kill thread')
                break
            if self.stopped or not self.grabbed:
                continue
            # -----------------model info-----------------
            # frame = cv2.resize(frame, (1280, 720))  #(800, 450)
            yolo_frame, lable, bbox = self.yolo.yolo(frame[..., ::-1])
            yolo_frame = cv2.resize(yolo_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            image = qimage2ndarray.array2qimage(yolo_frame)

            # -----------------video info-----------------
            if self.record and self.video_writer == False:
                self.video_writer = True
                self.polt_setting = [0] *12
                self.polt_spike = [0] *12
                self.polt_block = [0] *12

            if self.record and self.video_writer == True:
                print('recording')
                if 0 in lable:
                    self.polt_block.append(0.2)
                else:
                    self.polt_block.append(0)
                if 6 in lable:
                    self.polt_spike.append(0.5)
                else:
                    self.polt_spike.append(0)
                if 5 in lable:
                    self.polt_setting.append(1)
                else:
                    self.polt_setting.append(0)


            if self.re and self.video_writer == True:
                self.video_writer = False
                self.record = False
                self.re = False
                self.polt_setting = self.polt_setting + [0] * 12
                self.polt_spike = self.polt_spike + [0] * 12
                self.polt_block = self.polt_block + [0] * 12

                self.polt_setting = self.process_label(self.polt_setting)
                self.polt_spike = self.process_label(self.polt_spike)
                self.polt_block = self.process_label(self.polt_block)
                self.polt_win = [0] * len(self.polt_setting)
                if len(self.polt_setting) > 45:
                    self.polt_win[-45] = 1

                action_sum = [x + y for x, y in zip(self.polt_setting, self.polt_spike)]
                num, rang = find_continuous_blocks_with_ranges(action_sum)
                end_point = len(self.polt_setting) - 45


                # is_best = False
                # start = 0
                # if num == 1 and list(rang[0])[1] < end_point and list(rang[0])[0] + 40 < end_point:
                #     is_best = True
                #     start = list(rang[0])[0]
                # if num > 1 and list(rang[-1])[1] < end_point and list(rang[-1])[0] + 40 < end_point:
                #     is_best = True
                #     start = list(rang[-1])[0]
                # elif num >1 and list(rang[-2])[1] < end_point and list(rang[-2])[0] + 40 < end_point:
                #     is_best = True
                #     start = list(rang[-2])[0]

                is_best = False
                start = 0
                if num == 1 and list(rang[0])[0] + 40 < end_point:
                    is_best = True
                    start = list(rang[0])[0]
                if num > 1 and list(rang[-1])[0] + 40 < end_point:
                    is_best = True
                    start = list(rang[-1])[0]
                elif num >1 and list(rang[-2])[0] + 40 < end_point:
                    is_best = True
                    start = list(rang[-2])[0]


                if is_best:
                    Process(target=check_file, args=(start, end_point, len(self.polt_setting))).start()


                plt.figure(figsize=(self.diagram.width() / 80, self.diagram.height() / 50), dpi=300)
                plt.plot(self.polt_block, color='red')
                plt.plot(self.polt_spike, color='blue')
                plt.plot(self.polt_setting, color='green')
                plt.plot(self.polt_win, color='black')

                plt.legend()
                # plt.xticks([])
                plt.yticks([])
                # save = 'save_video/yolo/' + str(len(os.listdir('save_video/yolo'))) + '.png'
                plt.savefig('out.png')
                # save_array = {'block': self.polt_block, 'spike': self.polt_spike, 'setting': self.polt_setting, 'win': self.polt_win}
                # with open((save[:-4] + '.json').replace('yolo', 'array'), 'w') as f:
                #     json.dump(save_array, f)
                img = cv2.imread('out.png')[80:-80, 290:-230, :]
                img = cv2.resize(img, (self.diagram.width(), self.diagram.height()), interpolation=cv2.INTER_AREA)
                if is_best:
                    cv2.putText(img, 'save_best', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                img = qimage2ndarray.array2qimage(img)
                self.diagram.setPixmap(QtGui.QPixmap.fromImage(img))
                # save label
            # -----------------video info-----------------
            self.qlabel.setPixmap(QtGui.QPixmap.fromImage(image))


    def kill_thread(self):
        self.kill = True
        self.stream.release()

    def process_label(self, label_array):
        new = []
        for x in range(len(label_array) - 25):
            res = np.array(label_array[x:x + 25])
            out = sum(res * self.arr)
            new.append(out)
        return new




#
