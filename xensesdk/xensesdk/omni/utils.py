import numpy as np


def make_neighbors(vertices, indices):
    n = len(vertices)
    neighbors = [[] for _ in range(n)]

    for quad in indices:
        for i in range(3):
            neighbors[quad[i]].extend([quad[(i+1)%3], quad[(i+2)%3]])

    # 去重邻居顶点
    for i in range(n):
        neigh = list(set(neighbors[i]))
        if len(neigh) > 6:
            neigh = neigh[:6]
        elif len(neigh) < 6:
            neigh = [i] * 6
        # neigh.append(i)
        neighbors[i] = neigh
    return np.array(neighbors, np.int32)

def smooth(neighbors, vert_wise_data):
    """
    对顶点数据进行平滑, 每个节点的平滑后的数据为其邻居节点数据的平均值

    Parameters:
    - neighbors : np.ndarray of shape(n, m), n 为节点数, m 为每个节点的邻居数
    - vert_wise_data : np.ndarray of shape(n, k), k 为每个节点的数据维度
    """
    # n*3 -> 1*n*3, n*4 -> n*4*1 --> n*4*3
    extracted_normals = np.take_along_axis(vert_wise_data[None, ...], neighbors[..., None], axis=1)
    new_data = np.mean(extracted_normals, axis=1, keepdims=False)
    return new_data
