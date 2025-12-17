from __future__ import annotations  # Python 3.7+
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .DAGRunner import DAGBase 
import cv2
from .dataProcessor import *
from xensesdk.zeroros.timer import Timer

class AutoRefresher:
    def __init__(self, dag_runner:"DAGBase"):
        self.dag_runner = dag_runner
        refresh_time = 30 # s
        self.press_min_thresh = 0.99999
        self.timer = Timer(refresh_time*1000, self.refreshReference, False, delay_ms=1000)
        self.original_ref = None

    def refreshReference(self):
        if self.dag_runner.reference_image is not None:
            try:
                if self.original_ref is None:
                    self.original_ref = self.dag_runner.reference_image 
                    
                frame = self.dag_runner.sensor._real_camera.get_frame()[1]
                img_float = convertUint8ToInfer(frame, self.dag_runner.sensor.infer_size)
                img_ref = convertMarkerFree(self.dag_runner.sensor._infer_engine, img_float)
                diff_max = np.clip((img_ref - self.original_ref) * 2.5 + 110 / 255.0, 0, 1).max()
                # ssim = self.calSSIM(self.original_ref, img)
                # print(ssim)
                # print(ssim, diff_max)
                if diff_max < 0.5:
                    self.dag_runner.resetRefernceImage(image_float=img_float, image_ref=img_ref)
                    # print("reset ref")
            except Exception as e :
                print(f"Auto refresher error:{e}")
                # print("ref checking exit")

    def calSSIM(self, ref_img, cur_img):
        # 参数，遵循SSIM原论文
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # 高斯滤波，模拟局部统计特性
        kernel_size = 11
        sigma = 1.5
        window = cv2.getGaussianKernel(kernel_size, sigma)
        window = window @ window.T  # 生成二维高斯核

        # 均值（mu）
        mu1 = cv2.filter2D(ref_img, -1, window)
        mu2 = cv2.filter2D(cur_img, -1, window)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # 方差（sigma）
        sigma1_sq = cv2.filter2D(ref_img * ref_img, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(cur_img * cur_img, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(ref_img * cur_img, -1, window) - mu1_mu2

        # SSIM计算公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 返回均值作为整体得分
        return ssim_map.mean()
    
    def start(self):
        self.timer.start()

    def release(self):
        self.timer.stop()