from xensesdk.xenseInterface.XenseSensor import Sensor
 # @profile
img_cap = Sensor.create(0, use_gpu=False)

for i in range(300):
    img_cap.getAllSensorInfo()
   