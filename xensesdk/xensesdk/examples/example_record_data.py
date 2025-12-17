from xensesdk import Sensor
import time 

if __name__ == '__main__':
    sensor  = Sensor.create("OG000023")

    sensor.startSaveSensorInfo(r"/home/czl/Downloads/workspace/xensesdk/data/record_exp", [Sensor.OutputType.Difference, Sensor.OutputType.Rectify])
    time.sleep(5)
    sensor.stopSaveSensorInfo()
    print("save ok")
    
    sensor.release()
