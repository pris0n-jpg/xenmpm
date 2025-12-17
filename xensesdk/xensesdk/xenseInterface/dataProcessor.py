

import cv2
import numpy as np


def convertMarkerToFem(marker, grid_coord_size):
    physical_marker = np.array(marker, np.float32)
    physical_marker[..., 0] = grid_coord_size[0] - physical_marker[...,0]
    physical_marker = np.flip(physical_marker, 1)
    return physical_marker


def convertUint8ToInfer(img_uint8, infer_size):
    img_uint8 = img_uint8.astype(np.float32) / 255.0
    img_uint8 = cv2.resize(img_uint8, infer_size)
    return img_uint8
    
def convertOriginSize(img, src_size):
    return cv2.resize(img, src_size)

def convertMarkerFree(infer_engine, img_float):
    '''
    如果有marker则移除marker
    img_float: shape [infer_size, 3] float32, 
    '''
    img_marker_free = infer_engine.inferDiff(img_float)
    return img_marker_free

def convertObjectEnhance(img_marker_free, ref_img):
    '''
    找出img与_ref_img间差异, 使差异部分更明显
    '''
    img_obj_enhance = np.clip((img_marker_free - ref_img) * 2.5 + 110 / 255.0, 0, 1) 
    return img_obj_enhance

def convertMarkerEnhance(img_float, img_marker_free):
    '''
    斑点二值化
    '''
    img_marker_enhance = (img_float - img_marker_free) * 2
    img_deform_binary = img_marker_enhance <= -0.025
    img_deform_binary = np.all(img_deform_binary, axis=2, keepdims=True).astype(np.float32)
    
    return img_deform_binary

def convertDepth(infer_engine, img_obj_enhance):
    '''
    返回深度图像
    '''
    img_depth = infer_engine.inferDepth(img_obj_enhance)
    return img_depth

def predictMarker(infer_engine, img_marker_enhance, img_ref_binary):
    """
    预测 marker 位置和置信度
    """
    grid_conf = infer_engine.inferMarker(img_marker_enhance, img_ref_binary)
    return grid_conf

def detectMarker(img_marker_enhance, min_area, grid_coord_size):
    img = cv2.resize(img_marker_enhance, grid_coord_size)
    contours, bboxes, centers, areas = _extractMarkerRegions(img, min_area)

    return np.array(centers)

def _extractMarkerRegions(img_marker_enhance, min_area=60):
    """
    提取斑点区域的: 轮廓、bbox、中心、面积、掩膜

    Parameters:
    - image : np.ndarray
    - min_area : int, optional, default: 40, 最小面积
    """
    # 创建掩膜，提取符合颜色范围的区域
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(image, lower_thresh, 255)
    img_marker_enhance = (img_marker_enhance * 255).astype(np.uint8)
    # 查找斑点轮廓
    contours, _ = cv2.findContours(img_marker_enhance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算斑点 bbox 面积、中心, 并且排除掉面积为 min_area 的斑点
    bboxes = np.array([cv2.boundingRect(contour) for contour in contours], dtype=np.int32)
    if len(bboxes) == 0:
        return None, None, [[0, 0]], None, None

    areas = np.array([w*h for w, h in bboxes[:, 2:]])

    # 滤除小面积斑点以及比例不合适的斑点
    filter_mask = (areas > min_area) & (bboxes[:, 2] / bboxes[:, 3] < 4) & (bboxes[:, 2] / bboxes[:, 3] > 0.25)

    bboxes = bboxes[filter_mask]
    areas = areas[filter_mask]
    centers = [(x+w//2, y+h//2) for x, y, w, h in bboxes]

    return contours, bboxes, centers, areas

