import sys
import time
import numpy as np
import cv2 as cv
import NDIlib as ndi

def main():

    if not ndi.initialize():
        return 0

    cap = cv.VideoCapture('/home/ubuntu/Desktop/final/result1/0.mp4')

    send_settings = ndi.SendCreate()
    send_settings.ndi_name = '107'

    ndi_send = ndi.send_create(send_settings)

    video_frame = ndi.VideoFrameV2()

    start = time.time()
    while True:
        start_send = time.time()

        for _ in reversed(range(200)):

            ret, img = cap.read()

            if ret:
                img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

                video_frame.data = img
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

                ndi.send_send_video_v2(ndi_send, video_frame)

            else:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        print('200 frames sent, at %1.2ffps' % (200.0 / (time.time() - start_send)))

    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0

if __name__ == "__main__":
    sys.exit(main())