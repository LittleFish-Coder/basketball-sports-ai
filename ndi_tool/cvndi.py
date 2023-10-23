import os
import NDIlib as ndi
import time
import cv2 as cv
import numpy as np



def get_sources():
    if not ndi.initialize():
        return 0
    ndi_find = ndi.find_create_v2()
    if ndi_find is None:
        return 0
    sources = []
    t = time.time()
    while time.time() - t < 1.0 * 3:
        print('Looking for sources ...')
        ndi.find_wait_for_sources(ndi_find, 1000)
        sources = ndi.find_get_current_sources(ndi_find)
    for s in sources:
        print(s.ndi_name)

    return sources



def ip_source(sources, ip):
    for s in sources:
        if str(ip) in s.ndi_name:
            return s
    return None



class VideoCapture():
    def __init__(self, sources):
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        ndi.recv_connect(self.ndi_recv, sources)

    def read(self):
        t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 1000)
        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.ndi_recv, v)
            return True, frame
        else:
            return False, None
#
    def release(self):
        ndi.recv_destroy(self.ndi_recv)
        ndi.destroy()
