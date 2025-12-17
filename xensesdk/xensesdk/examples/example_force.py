import sys
from xensesdk import ExampleView
from xensesdk import Sensor
import cv2

flag = 1

def main():
    sensor_0 = Sensor.create(0)
    View = ExampleView(sensor_0)
    View2d = View.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Depth, Sensor.OutputType.Marker2D)
    # View2d = View.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Marker2D, Sensor.OutputType.Marker2D)

    def callback():
        force, res_force, mesh_init, rectify_real, diff, depth = sensor_0.selectSensorInfo(
            Sensor.OutputType.Force, 
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Rectify, 
            Sensor.OutputType.Difference, 
            Sensor.OutputType.Depth,
        )

        marker_img = sensor_0.drawMarkerMove(rectify_real)
        View2d.setData(Sensor.OutputType.Marker2D, marker_img)

        View2d.setData(Sensor.OutputType.Difference, diff)
        View2d.setData(Sensor.OutputType.Depth, depth)
        View.setForceFlow(force, res_force, mesh_init)
        View.setDepth(depth)
        

        # 获取其他信息
        marker, mesh = sensor_0.selectSensorInfo(Sensor.OutputType.Marker2D,Sensor.OutputType.Mesh3DInit)
        global flag 
        if flag == 1:
            print('force',force.shape,
                    '\nres_force',res_force.shape,
                    '\nmesh_init',mesh_init.shape,
                    '\nrectify_real',rectify_real.shape,
                    '\ndiff',diff.shape,
                    '\ndepth',depth.shape,
                    '\nmarker',marker.shape,
                    # '\nmarker_x',marker[:,:,0],
                    '\nmesh',mesh.shape,
                    # '\nmesh_x',mesh[:,:,1]
                    )
            flag = 0
        # print('\nmarker_x',marker[:,:,0],)

    View.setCallback(callback)
    View.show()
    sensor_0.release()
    sys.exit()

if __name__ == '__main__':
    main()