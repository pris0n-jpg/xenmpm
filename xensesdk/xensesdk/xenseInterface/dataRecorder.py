import time
import h5py
from pathlib import Path
import numpy as np
import cv2
from .sensorEnum import OutputType
from xensesdk.ezgl.utils.video_utils import CV2VideoWriter
from xensesdk.zeroros.timer import Timer

class DataRecorder:
    def __init__(self, sensor):
        self.sensor = sensor
        self.timer = None
        self.h5file = None
        self.data_to_save = None
        self.video_writers = {}

    def startRecord(self, path, data_to_save):
        self.timer = Timer(1000/30, self.collectionLoop, False) # 30hz 
        self.initDataPath(path, data_to_save)
        self.timer.start()
    
    def stopRecord(self):
        if self.timer:
            self.timer.stop()
        if self.video_writers:
            for writer in self.video_writers.values():
                writer.release()
        if self.h5file:
            self.h5file.close()


    def initDataPath(self, path, data_to_save):
        self.data_to_save = data_to_save
        path = Path(path) if isinstance(path, str) else path

        # 格式化日期和时间（年月日时分秒）
        formatted_datetime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

        # Initialize HDF5 file and resources
        self.h5file = h5py.File(path / f"sensor_{self.sensor._real_camera._cam_id}_stamped_data_{formatted_datetime}.h5", "w")
        self.video_writers = {}

        video_settings = {
            # OutputType.Raw: {"filename": "raw_video"},
            OutputType.Difference: {"filename": "diff_video" },
            OutputType.Depth: {"filename": "depth_video" }
        }
        
        for output_type, config in video_settings.items():
            if output_type in data_to_save:
                self.video_writers[output_type] = CV2VideoWriter(
                    path / f"sensor_{self.sensor._real_camera._cam_id}_{config['filename']}_{formatted_datetime}.mp4",
                    (400, 700)
                )
        
        if OutputType.Marker2D in data_to_save:
            max_shape = (self.sensor._config_manager.marker_config.nrow, self.sensor._config_manager.marker_config.ncol, None)
            self.h5file.create_dataset("marker_data", shape=(self.sensor._config_manager.marker_config.nrow, self.sensor._config_manager.marker_config.ncol, 0),
                                maxshape=max_shape, dtype="float32")
        if OutputType.Rectify in data_to_save:
            self.h5file.create_dataset('rectify_data', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype('uint8')))

        self.h5file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.dtype([('timestamp_ns', np.int64), ('frame_type', 'S1')]))

    def collectionLoop(self):
        fetched_data = self.sensor.selectSensorInfo(*self.data_to_save)
        curr_time = time.time()

        for key, writer in self.video_writers.items():
            if key in self.data_to_save:
                if len(self.data_to_save) == 1:
                    frame = fetched_data
                else:
                    frame = fetched_data[self.data_to_save.index(key)]
                if key == OutputType.Depth:
                    frame = (frame * 170).astype(np.uint8)
                if key == OutputType.Difference:
                    frame = frame.astype(np.uint8)
                writer.write(frame)

        if OutputType.Rectify in self.data_to_save:
            dataset = self.h5file["rectify_data"]
            if len(self.data_to_save) == 1:
                frame = fetched_data
            else:
                frame = fetched_data[self.data_to_save.index(OutputType.Rectify)]
            _, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            temp = frame[:100].copy()
            frame[:100] = frame[100:200]
            frame[100:200] = temp
            current_size = dataset.shape[0]
            dataset.resize((current_size + 1,))
            dataset[current_size] = frame

        if OutputType.Marker2D in self.data_to_save:
            dataset = self.h5file["marker_data"]
            current_size = dataset.shape[2]
            marker_data = fetched_data[self.data_to_save.index(OutputType.Marker2D)]
            new_size = current_size + marker_data.shape[2]
            dataset.resize((dataset.shape[0], dataset.shape[1], new_size))
            dataset[:, :, current_size:new_size] = marker_data

        ts_ds = self.h5file["timestamps"]
        current_size = ts_ds.shape[0]
        ts_ds.resize((current_size + 1,))
        ts_ds[current_size] = np.array([(curr_time*1e9, "I")], dtype=np.dtype([('timestamp_ns', np.int64), ('frame_type', 'S1')]))
    
    def isRecording(self):
        if self.timer:
            return self.timer.alive()
        else:
            return False