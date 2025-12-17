from xensesdk.xenseInterface.XenseSensor import Sensor
from pathlib import Path

def get_files_in_directory(directory_path):
    path = Path(directory_path)
    return [file for file in path.rglob('*') if file.is_file()]

# 使用例子

if __name__ == '__main__':
    sensor = Sensor.create(0, check_serial=False)
    directory_path = r'D:\gitlab\xensesdk\xensesdk\examples\config'
    files = get_files_in_directory(directory_path)
    for src_path in files:

        dst_path = Path(r'D:\gitlab\xensesdk\xensesdk\examples\config_fine') / src_path.stem
        sensor.loadConfig(str(src_path))
        marker_config = sensor.fetchMarkerConfig()
        rectify_config = sensor.fetchRectifyConfig()
        marker_config['ncol'] = 11
        marker_config['nrow'] = 20
        sensor.resetMarkerConfig(marker_config)
        sensor.saveConfig(str(dst_path))
