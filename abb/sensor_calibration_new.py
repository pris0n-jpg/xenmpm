import cv2
import sys
import os
import time
from time import sleep
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured
from datetime import datetime
from threading import Thread
from pathlib import Path
from PyQt5 import QtWidgets
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator as RGI
import yaml

import matplotlib.pyplot as plt

from pyabb import ABBRobot, Logger, Affine
from pyati.ati_sensor import ATISensor
logger = Logger(log_level='DEBUG', name="Main", log_path=None)
from log_writer import LogWriter
from xensesdk import Sensor


PROJ_DIR = Path(__file__).resolve().parent.parent # task文件夹绝对路径

TIME_STAMP = str(datetime.now().strftime('%y_%m_%d__%H_%M_%S'))


class TactileSensor():
    def __init__(self):
        """
        manage a pair of xense tactile sensors
        """
        self.sensor = Sensor.create(2)

        ## data varible
        self.sensor_node_index = np.arange(35 * 20, dtype=int).reshape((35, 20), order='F')  # connection
        self.sensor_quad_surface = np.zeros((34 * 19, 4), dtype=int)  # first y, than x
        self.sensor_triangle_surface = np.zeros((34 * 19 * 2, 3), dtype=int)
        for j in range(19):
            for i in range(34):
                self.sensor_quad_surface[j*34 + i, :] =np.array([self.sensor_node_index[i, j], self.sensor_node_index[i+1, j], self.sensor_node_index[i+1, j+1], self.sensor_node_index[i, j+1]])
                self.sensor_triangle_surface[2*(j*34 + i), :] = np.array([self.sensor_quad_surface[j*34 + i, 0], self.sensor_quad_surface[j*34 + i, 1], self.sensor_quad_surface[j*34 + i, 2]])
                self.sensor_triangle_surface[2*(j*34 + i) + 1, :] = np.array([self.sensor_quad_surface[j*34 + i, 0], self.sensor_quad_surface[j*34 + i, 3], self.sensor_quad_surface[j*34 + i, 2]])

        self.sensor_mesh = None
        self.sensor_force = None
        self.sensor_res_force = None
        self.sensor_points = None

    def get_data(self):
        force, res_force ,mesh_flow = self.sensor.selectSensorInfo(
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DFlow
        )

        return force, res_force ,mesh_flow

    def release(self):
        self.sensor.release()


class ServoController():
    """
    移动到任意像素坐标
    采集图片
    采集传感器数据
    下压
    """
    def __init__(self, pose0,
                 log_dir=PROJ_DIR / f"data/img_servo/{TIME_STAMP}_square"):
        ## 机器人初始化
        # 启动并连接至机器人服务
        self.robot = ABBRobot(ip="192.168.125.1",
             port_motion = 5000,
             port_logger = 5001,
             port_signal = 5002,)
        logger.warning("Connect to Server")
        self.robot.initialize()
        # 移动到初始位置
        self.pose0 = pose0
        self.robot.set_acceleration(0.5, 0.5)
        self.robot.set_velocity(20, 20)
        self.robot.moveCart(self.pose0)
        self.checkJointLimit() # 检查末端关节角是否接近超限
        time.sleep(1)
        print("init pose: ", self.robot.get_cartesian())
        print("init velocity: ", self.robot.get_velocity())

        # ATI传感器初始化
        self.ati = ATISensor(ip="192.168.1.10", filter_on=False)
        time.sleep(2)
        self.ati.tare()

        # xense 初始化
        self.sensor = TactileSensor()

        ## 记录初始化
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        filename = "log_" + TIME_STAMP + ".csv"
        self.log_writer = LogWriter(self.log_dir/filename,
                                    fields=["id","x_mm", "y_mm", "z_mm",'F_x','F_y','F_z',"indent_mm", 'contact_node_num', 'ux', 'uy', 'uz', 'ux^mm', 'uy^mm', 'uz^mm'])

        z_cont = -0.01
        cont_th1 = -0.03
        cont_th2 = -0.03001
        self.z_cont = z_cont#-38
        self.cont_th1 = cont_th1#-0.02
        self.cont_th2 = cont_th2#-0.05


    def get_robot_xyz(self):
        return self.robot.get_cartesian().x, self.robot.get_cartesian().y, self.robot.get_cartesian().z

    def get_ati(self):
        return self.ati.data

    def move_to_xyz(self, x, y, z):
        cp = self.robot.get_cartesian()
        target_pose = Affine(x=x, y=y, z=z, a=cp.a, b=cp.b, c=cp.c)
        self.robot.moveCart(target_pose)

    def relative_move(self, x=0, y=0, z=0, Rz=0, Ry=0, Rx=0):
        cp = self.robot.get_cartesian()
        target_pose = Affine(x=cp.x, y=cp.y, z=cp.z, a=cp.a, b=cp.b, c=cp.c) * Affine(x=x, y=y, z=z, a=Rz, b=Ry, c=Rx)  # avoid to change current pose
        self.robot.moveCart(target_pose)

    def move0(self,x,y,z,qw0,qx0,qy0,qz0,velocity):
        """不带接触检测的移动"""
        # xy 相对移动
        self.robot.set_velocity(velocity, velocity)
        x00 = self.robot.get_cartesian().x
        y00 = self.robot.get_cartesian().y
        z00 = self.robot.get_cartesian().z
        self.pose1=[x00+x,y00+y,z00,qw0,qx0,qy0,qz0]
        self.robot.moveCart(self.pose1)
        time.sleep(0.2)
        print("move to xy_pose: ", self.robot.get_cartesian())
        print("move xy_velocity: ", self.robot.get_velocity())
        # z移动
        self.robot.set_velocity(velocity, velocity)
        x01 = self.robot.get_cartesian().x
        y01 = self.robot.get_cartesian().y
        z01 = self.robot.get_cartesian().z
        self.pose2=[x01,y01,z01+z,qw0,qx0,qy0,qz0]
        self.robot.moveCart(self.pose2)
        time.sleep(0.2)
        print("move to z_pose: ", self.robot.get_cartesian())
        print("move z_velocity: ", self.robot.get_velocity())

    def rotate_quaternion(self,q1,q2,q3,q4):
        """不带接触检测的旋转"""
        # 相对转动
        xr0 = self.robot.get_cartesian().x
        yr0 = self.robot.get_cartesian().y
        zr0 = self.robot.get_cartesian().z
        self.pose1=[xr0,yr0,zr0,q1,q2,q3,q4]
        self.robot.moveCart(self.pose1)
        # time.sleep(0.2)
        print("move to pose: ", self.robot.get_cartesian())
        print("move velocity: ", self.robot.get_velocity())

    def down_util_contact(self):
        is_contact = False

        # 安全检测
        fz = self.get_ati()[2]
        if fz <= -10:
            print('力过大，退出')
            exit()
        while not is_contact:
            fz1 = self.get_ati()[2]
            print(f'ATI force: {fz1}, is < cont_th1: {fz1 <= self.cont_th1}, is< cont_th2: {fz1 <= self.cont_th2}')
            # if fz1 <= self.cont_th1:
            #     # 再向下 0.02 确认是否接触
            #     pose_t=[self.robot.get_cartesian().x,self.robot.get_cartesian().y,self.robot.get_cartesian().z-0.02,
            #             self.robot.get_cartesian().q_w,self.robot.get_cartesian().q_x,self.robot.get_cartesian().q_y,self.robot.get_cartesian().q_z]
            #     self.robot.moveCart(pose_t)
            #     time.sleep(1)
            #     fz2 = self.get_ati()[2]
            #     if fz2 <= self.cont_th2:
            #         pose_t=[self.robot.get_cartesian().x,self.robot.get_cartesian().y,self.robot.get_cartesian().z+0.02,
            #                 self.robot.get_cartesian().q_w,self.robot.get_cartesian().q_x,self.robot.get_cartesian().q_y,self.robot.get_cartesian().q_z]
            #         self.robot.moveCart(pose_t)
            #         time.sleep(1)
            #         self.z_cont = self.robot.get_cartesian().z
            #         print(f"contact detected. z_cont: {self.z_cont}mm | fz1: {fz1} | fz2: {fz2}")
            #         is_contact = True
            #         break

            if fz1 <= self.cont_th2:
                self.z_cont = self.robot.get_cartesian().z
                print(f"contact detected. z_cont: {self.z_cont}mm | fz3: {fz1}")
                is_contact = True
                break

            # move down slowly
            self.relative_move(z=0.02)
            time.sleep(0.5)
            print(f'Moved to {self.robot.get_cartesian()}')

    def checkJointLimit(self):
        current_joint = self.robot.get_joint()
        if current_joint[5] > 180:
            self.robot.moveJoint(current_joint[0],current_joint[1],current_joint[2],
                             current_joint[3],current_joint[4],current_joint[5] - 360)
        elif current_joint[5] < -180:
            self.robot.moveJoint(current_joint[0],current_joint[1],current_joint[2],
                             current_joint[3],current_joint[4],current_joint[5] + 360)

    def getRelativePose(self,current_pose,object_pose) -> list:
        relative_pose = [float(current_pose.x) - float(object_pose.x),float(current_pose.y) - float(object_pose.y),float(current_pose.z) - float(object_pose.z),
                         float(current_pose.c) - float(object_pose.c),float(current_pose.b) - float(object_pose.b),float(current_pose.a) - float(object_pose.a)]
        return relative_pose

    def getAbsolutePose(self,relative_pose,object_pose) -> list:
        absolute_pose = [relative_pose[0]+object_pose[0],relative_pose[1]+object_pose[1],relative_pose[2]+object_pose[2],
                         relative_pose[3]+object_pose[3],relative_pose[4]+object_pose[4],relative_pose[5]+object_pose[5]]
        return absolute_pose

    def affine2ABBEulerPose(self,affine):
        return [affine.x, affine.y, affine.z, affine.a, affine.b, affine.c]

    def servoPose2ABBPose(self, servo_pose):
        return [servo_pose[0],servo_pose[1],servo_pose[2],servo_pose[5],servo_pose[4],servo_pose[3]]

    def ABBPose2ServoPose(self, abb_pose):
        return [abb_pose[0],abb_pose[1],abb_pose[2],abb_pose[5],abb_pose[4],abb_pose[3]]

if __name__ == '__main__':
    log_dir = PROJ_DIR / f"collector/data/sensor_calibration_cube/sensor_0_soft_new"

    # 机械臂和传感器初始化
    # 机器人运动到指定位置0（中立初始位置）取决于任务需要
    # pose0=[559,-199.9,110,0,1,0,0]  # ATIsensor & 圆压头
    # pose0=[550.64,-45.3,189.5 + 20,0,1,0,0] # 方形平台
    # pose0 = [566.55, -200.6, 85 + 20, 0, 1, 0, 0] # 与ATIsensor & 物块 (cube) 接触
    pose0 = [574.33, -176.67, 94.89 + 50, 0, 1, 0, 0]



    sc = ServoController(pose0=pose0,log_dir=log_dir)
    sc.robot.set_acceleration(0.1, 0.1)
    sc.robot.set_velocity(10, 10)
    sc.sensor.get_data() # 传感器初始化

    # sc.move_to_xyz(x=559, y=-199.9, z=94.88 + 0.3) # ATIsensor & 圆压头
    # sc.move_to_xyz(x=550.64, y=-45.3, z=189.5) # 方形平台
    # sc.move_to_xyz(x=566.55, y=-200.6, z=85) # ATIsensor & 物块接触
    # 0919 ATI + 立方体 + 宽15的片
    corner_coord = np.array([574.33, -176.67, 94.89])
    sc.move_to_xyz(x=corner_coord[0], y=corner_coord[1] + (17.5 - 15)/2, z=corner_coord[2] + 5)

    time.sleep(1)
    print('down contact begin')
    sc.down_util_contact()

    indent_num = 15
    dz = 0.05
    for i in np.arange(indent_num):
        x, y, z = sc.get_robot_xyz()
        indent = sc.z_cont - z

        frame_num = 20
        period = 0.2  # second
        sensor_force, sensor_res_force, sensor_mesh_flow = sc.sensor.get_data()
        ati_data = sc.get_ati().copy()

        for j in np.arange(frame_num):
            sensor_force_new, sensor_res_force_new, sensor_mesh_flow_new = sc.sensor.get_data()
            sensor_force += sensor_force_new
            sensor_res_force += sensor_res_force_new
            sensor_mesh_flow += sensor_mesh_flow_new

            ati_data += sc.get_ati().copy()
            time.sleep(period)

        sensor_force /= (frame_num+1)
        sensor_res_force /= (frame_num+1)
        sensor_mesh_flow /= (frame_num+1)
        ati_data /= (frame_num+1)

        contact_area = np.where(sensor_mesh_flow[:,:,2] < np.min(sensor_mesh_flow[:,:,2])/2)
        contact_disp = sensor_mesh_flow[contact_area[0],contact_area[1],:]
        average_disp = np.sum(contact_disp, axis=0) / contact_area[0].size
        # average_disp = np.sum(sensor_mesh_flow, axis=(0,1)) / sensor_mesh_flow[:,:,2].size

        contact_disp_square = sensor_mesh_flow[contact_area[0],contact_area[1],:] ** 2
        average_disp_square = np.sum(contact_disp_square, axis=0) / contact_area[0].size

        contact_force = sensor_force[contact_area[0],contact_area[1],:]
        average_force = np.sum(contact_force, axis=0) / contact_area[0].size
        contact_node_num = contact_area[0].size

        print(f'F: {ati_data.tolist()} , U: {average_disp.tolist()}, contact_num: {contact_node_num}')

        sc.log_writer.log(i, x, y, z, ati_data[0], ati_data[1], ati_data[2], indent, contact_node_num, average_disp[0], average_disp[1], average_disp[2], average_disp_square[0], average_disp_square[1], average_disp_square[2], echo=True)
        sc.relative_move(z=dz)
        time.sleep(1)

    sc.relative_move(z=-1.5 - indent_num * dz)
    sc.sensor.release()
    # 停止机器人马达
    time.sleep(1)
    sc.robot.sig_motor_off()

