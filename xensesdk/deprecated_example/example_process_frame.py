from xensesdk.ezgl.utils.QtTools import qtcv
from xensesdk.xenseInterface.XenseSensor import Sensor
import numpy as np

#################################################
### this script shows how to do image process ###
#################################################
def getframeExample(camera_id, use_gpu):
    omnisensor = Sensor.create(camera_id, use_gpu)

    while True:
        # get a frame from cam
        image = omnisensor.getRectifyImage()

        qtcv.imshow('src', image)
        if qtcv.waitKey(1) == qtcv.Key.Key_Q:  # 如果按下 'q' 键
            print("You pressed 'q' to quit.")
            break

def diffExample(camera_id, use_gpu):
    omnisensor = Sensor.create(camera_id, use_gpu)
    # this flag shows that you can either automatically get a frame to process or directly give a img to process
    flag = True
    while True:
        if flag:
            diff_image = omnisensor.getDiffImage()
            qtcv.imshow('src', np.ascontiguousarray(diff_image))
            flag = False
        else:
            src_img = omnisensor.getRectifyImage()
            diff_image = omnisensor.getDiffImage(src_img)
            qtcv.imshow('src', np.ascontiguousarray(diff_image))
            flag = True

        if qtcv.waitKey(1) == qtcv.Key.Key_Q:  # 如果按下 'q' 键
            print("You pressed 'q' to quit.")
            break

def depthExample(camera_id, use_gpu):
    omnisensor = Sensor.create(camera_id, use_gpu)
    # this flag shows that you can either automatically get a frame to process or directly give a img to process
    flag = True
    while True:
        if flag:
            depth = omnisensor.getDepthImage() * 150
            qtcv.imshow('src', np.ascontiguousarray(depth))
            flag = False
        else:
            src_img = omnisensor.getRectifyImage()
            depth = omnisensor.getDepthImage(src_img) * 150
            qtcv.imshow('src', np.ascontiguousarray(depth))
            flag = True

        if qtcv.waitKey(1) == qtcv.Key.Key_Q:  # 如果按下 'q' 键
            print("You pressed 'q' to quit.")
            break


def markertrackExample(camera_id, use_gpu):
    omnisensor = Sensor.create(camera_id, use_gpu)

    while True:
        src_img = omnisensor.getRectifyImage()
        current_marker_pos, confidence = omnisensor.getMarker(src_img)
        marked_img = omnisensor.drawMarkerMove(src_img)
        qtcv.imshow('src', np.ascontiguousarray(marked_img))

        if qtcv.waitKey(1) == qtcv.Key.Key_Q:  # 如果按下 'q' 键
            print("You pressed 'q' to quit.")
            break

def allInfoExample(camera_id, use_gpu):
    omnisensor = Sensor.create(camera_id, use_gpu)
    # this flag shows that you can either automatically get a frame to process or directly give a img to process
    flag = True
    while True:
        if flag:
            sensor_info = omnisensor.getAllSensorInfo()
            cur_marker_pos = sensor_info['marker']
            diff_image = sensor_info['diff_img']
            src_img = sensor_info['src_img']
            depth = sensor_info['depth_map'] *150
            marker_img = omnisensor.drawMarkerMove(src_img)
            qtcv.imshow('src', np.ascontiguousarray(src_img))
            qtcv.imshow('depth_map', np.ascontiguousarray(depth))
            qtcv.imshow('marker_img', np.ascontiguousarray(marker_img))
            qtcv.imshow('diff_img', np.ascontiguousarray(diff_image))
            flag = False
        else:
            src_img = omnisensor.getRectifyImage()
            sensor_info = omnisensor.getAllSensorInfo(src_img)
            cur_marker_pos = sensor_info['marker']
            diff_image = sensor_info['diff_img']
            src_img = sensor_info['src_img']
            depth = sensor_info['depth_map'] *150
            marker_img = omnisensor.drawMarkerMove(src_img)
            qtcv.imshow('src', np.ascontiguousarray(src_img))
            qtcv.imshow('depth_map', np.ascontiguousarray(depth))
            qtcv.imshow('marker_img', np.ascontiguousarray(marker_img))
            qtcv.imshow('diff_img', np.ascontiguousarray(diff_image))
            flag = True

        if qtcv.waitKey(1) == qtcv.Key.Key_Q:  # 如果按下 'q' 键
            print("You pressed 'q' to quit.")
            break

if __name__ == '__main__':
    # getframeExample(camera_id = 0, use_gpu=True)
    # diffExample(camera_id = 0, use_gpu=True)
    # depthExample(camera_id = 0, use_gpu=True)
    # markertrackExample(camera_id = 0, use_gpu=True)
    allInfoExample(camera_id = 0, use_gpu=True)