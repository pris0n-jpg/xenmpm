import numpy as np


def compute_normals(vertices, indices):
    # 计算法向量
    # vertices: (n, 3) 顶点数组
    # indices: (m, 3) 三角形索引数组
    
    # 初始化法向量数组
    normals = np.zeros(vertices.shape, dtype=np.float32)

    # 获取三角形的三个顶点
    v0 = vertices[indices[:, 0]]  # (m, 3)
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    edge1 = v1 - v0  # (m, 3)
    edge2 = v2 - v0
    
    # 计算法向量
    normal = np.cross(edge1, edge2)  # (m, 3)
    
    # 将法向量加到每个顶点上
    np.add.at(normals, indices[:, 0], normal)
    np.add.at(normals, indices[:, 1], normal)
    np.add.at(normals, indices[:, 2], normal)
    
    # 对每个顶点的法向量进行归一化
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)  # 计算每个顶点的法向量的长度
    norm_len[norm_len < 1e-5] = 1  # 处理零向量
    normals /= norm_len  # 归一化每个顶点的法向量

    return normals

def compute_normals_quad(vertices, indices):
    # 计算四边形的法向量
    # vertices: (n, 3) 顶点数组
    # indices: (m, 4) 四边形索引数组
    
    # 初始化法向量数组
    normals = np.zeros(vertices.shape, dtype=np.float32)
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    v3 = vertices[indices[:, 3]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal1 = np.cross(edge1, edge2)
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal2 = np.cross(edge1, edge2)
    np.add.at(normals, indices[:, [0, 1, 2]], normal1[:, None])
    np.add.at(normals, indices[:, [1, 2, 3]], normal2[:, None])
    
    # 对每个顶点的法向量进行归一化
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)  # 计算每个顶点的法向量的长度
    norm_len[norm_len < 1e-5] = 1  # 处理零向量
    normals /= norm_len  # 归一化每个顶点的法向量

    return normals

def interp(mapxy, marker_grids, x_list, y_list, x_interval, y_interval):
    """
    计算插值

    Parameters:
    - mapxy : np.ndarray, float32, shape=(n, m, 2), cv2.remap 中的 mapx, mapy, 
    - marker_grids : np.ndarray, float32, shape=(p, q, 2), src 网格数据
    - x_list : np.ndarray, float32, shape=(p,), dst 规则网格 x 轴坐标列表, 等差数列
    - y_list : np.ndarray, float32, shape=(q,), dst 规则网格 y 轴坐标列表, 等差数列
    - x_interval : float, x 轴间隔
    - y_interval : float, y 轴间隔

    Returns:
    - mapxy : np.ndarray, float32, shape=(n, 2), 输出点数组
    """
    col_n = x_list.shape[0]
    row_n = y_list.shape[0]
    x_min = x_list[0]
    y_min = y_list[0]
    
    dst_h, dst_w = mapxy.shape[0], mapxy.shape[1]
    dst_x, dst_y = np.meshgrid(np.arange(dst_w), np.arange(dst_h))  # (h, w)    

    x_idx = np.clip(((dst_x - x_min) * (1 / x_interval)).astype(np.int32), 0, col_n - 2)  # (h, w)
    y_idx = np.clip(((dst_y - y_min) * (1 / y_interval)).astype(np.int32), 0, row_n - 2)  # (h, w)

    P11 = marker_grids[y_idx, x_idx]  # (h, w, 2)
    P12 = marker_grids[y_idx, x_idx + 1]
    P21 = marker_grids[y_idx + 1, x_idx]
    P22 = marker_grids[y_idx + 1, x_idx + 1]

    x1 = x_list[x_idx]  # (h, w)
    x2 = x_list[x_idx + 1]
    y1 = y_list[y_idx]
    y2 = y_list[y_idx + 1]

    u1 = (dst_x - x1)[..., None]
    u2 = (x2 - dst_x)[..., None]
    v1 = (dst_y - y1)[..., None]
    v2 = (y2 - dst_y)[..., None]

    mapxy[...] = (P11 * u2 * v2 + P12 * u1 * v2 + P21 * u2 * v1 + P22 * u1 * v1) / ((x2 - x1) * (y2 - y1))[..., None]


def inverse_map(mapx, mapy, imapx, imapy):
    """
    计算逆映射

    Parameters:
    - mapx : np.ndarray, float32, shape=(h, w), 输入的 x 坐标映射
    - mapy : np.ndarray, float32, shape=(h, w), 输入的 y 坐标映射
    - imapx : np.ndarray, float32, shape=(ih, iw), 输出的 x 坐标逆映射
    - imapy : np.ndarray, float32, shape=(ih, iw), 输出的 y 坐标逆映射
    """
    h, w = mapx.shape
    ih, iw = imapx.shape
    # 将 mapx 和 mapy 转换为整数索引
    mapx = mapx.astype(int)
    mapy = mapy.astype(int)

    # 创建一个掩码，确保索引在范围内
    mask = (mapx >= 0) & (mapx < iw) & (mapy >= 0) & (mapy < ih)  # (h, w)

    # 创建坐标网格
    y_indices, x_indices = np.indices((h, w))

    # 更新 imapx 和 imapy，根据掩码设置对应的值
    imapx[mapy[mask], mapx[mask]] = x_indices[mask]
    imapy[mapy[mask], mapx[mask]] = y_indices[mask]


# 示例数据
if __name__ == '__main__':
    pass

    # from ti_kernel import ti_interp, ti_inverse_map
    
    # x_list = np.array([0, 200, 400], dtype=np.float32)
    # y_list = np.array([10, 310, 610], dtype=np.float32)
    # marker_grids = np.array(
    #     [
    #         [[0, 0], [201, 1], [390, 2]],
    #         [[0, 230], [201, 230], [390, 230]],
    #         [[0, 500], [201, 500], [390, 500]],
    #     ], 
    #     dtype=np.float32
    # )
    # mapxy = np.zeros((700, 400, 2), dtype=np.float32)
    # mapxy1 = np.zeros((700, 400, 2), dtype=np.float32)

    # def test_interp():
    #     for i in range(100):
    #         ti_interp(mapxy, marker_grids, x_list, y_list, 100, 300)
    #         interp(mapxy1, marker_grids, x_list, y_list, 100, 300)
    #     print(np.allclose(mapxy, mapxy1))

    # def test_inverse_map():
    #     imapx, imapy = np.zeros((700, 400), dtype=np.float32), np.zeros((700, 400), dtype=np.float32)
    #     imapx1, imapy1 = np.zeros((700, 400), dtype=np.float32), np.zeros((700, 400), dtype=np.float32)
        
    #     for i in range(1000):
    #         mapxy = np.random.randint(0, 700, (100, 200, 2)).astype(np.float32)
    #         inverse_map(mapxy[:, :, 0], mapxy[:, :, 1], imapx, imapy)
    #         ti_inverse_map(np.ascontiguousarray(mapxy[:, :, 0]), np.ascontiguousarray(mapxy[:, :, 1]), imapx1, imapy1)
