import warnings

# Suppress any warning that contains the word "cupy" in the message
warnings.filterwarnings("ignore", message=".*cupy.*", category=UserWarning)
import cupy as cp
import numpy as np

cp.arange(1, dtype=np.float32)  # test if cuda is available

# ====================
# 三角面法向计算
# ====================
kernel_code_compute_normals = '''
extern "C" __global__
void compute_normals(const int* indices, const float* vertices, float* vertex_normals, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int idx1 = indices[i * 3];
    int idx2 = indices[i * 3 + 1];
    int idx3 = indices[i * 3 + 2];

    float3 v1 = make_float3(vertices[idx2 * 3] - vertices[idx1 * 3],
                            vertices[idx2 * 3 + 1] - vertices[idx1 * 3 + 1],
                            vertices[idx2 * 3 + 2] - vertices[idx1 * 3 + 2]);

    float3 v2 = make_float3(vertices[idx3 * 3] - vertices[idx1 * 3],
                            vertices[idx3 * 3 + 1] - vertices[idx1 * 3 + 1],
                            vertices[idx3 * 3 + 2] - vertices[idx1 * 3 + 2]);

    float3 normal = make_float3(v1.y * v2.z - v1.z * v2.y,
                                v1.z * v2.x - v1.x * v2.z,
                                v1.x * v2.y - v1.y * v2.x);

    atomicAdd(&vertex_normals[idx1 * 3], normal.x);
    atomicAdd(&vertex_normals[idx1 * 3 + 1], normal.y);
    atomicAdd(&vertex_normals[idx1 * 3 + 2], normal.z);
    atomicAdd(&vertex_normals[idx2 * 3], normal.x);
    atomicAdd(&vertex_normals[idx2 * 3 + 1], normal.y);
    atomicAdd(&vertex_normals[idx2 * 3 + 2], normal.z);
    atomicAdd(&vertex_normals[idx3 * 3], normal.x);
    atomicAdd(&vertex_normals[idx3 * 3 + 1], normal.y);
    atomicAdd(&vertex_normals[idx3 * 3 + 2], normal.z);
}
'''

# 编译自定义CUDA核函数
kernel_compute_normals = cp.RawKernel(kernel_code_compute_normals, 'compute_normals')

def cuda_compute_normals(vertices, indices):

    # 将数据转移到GPU上
    vertices_gpu = cp.asarray(vertices, dtype=cp.float32)
    indices_gpu = cp.asarray(indices)

    # 顶点数量和三角形数量
    n = vertices_gpu.shape[0]
    m = indices_gpu.shape[0]

    vertex_normals_gpu = cp.zeros((n, 3), dtype=cp.float32)

    # 执行自定义核函数
    threads_per_block = 64
    blocks_per_grid = (m + threads_per_block - 1) // threads_per_block
    kernel_compute_normals((blocks_per_grid,), (threads_per_block,),
                    (indices_gpu, vertices_gpu, vertex_normals_gpu, m))

    # 归一化顶点法线(np)
    vertex_normals = cp.asnumpy(vertex_normals_gpu)
    norm_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True) 
    norm_len[norm_len < 1e-5] = 1
    vertex_normals = vertex_normals / norm_len

    # 将结果从GPU拷贝回CPU
    return vertex_normals



# ====================
# 四面体法向计算
# ====================
kernel_code_compute_normals_quad = '''
extern "C" __global__
void compute_normals(const int* indices, const float* vertices, float* vertex_normals, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int idx1 = indices[i * 4];
    int idx2 = indices[i * 4 + 1];
    int idx3 = indices[i * 4 + 2];
    int idx4 = indices[i * 4 + 3];

    float3 v1 = make_float3(vertices[idx2 * 3] - vertices[idx1 * 3],
                            vertices[idx2 * 3 + 1] - vertices[idx1 * 3 + 1],
                            vertices[idx2 * 3 + 2] - vertices[idx1 * 3 + 2]);

    float3 v2 = make_float3(vertices[idx3 * 3] - vertices[idx1 * 3],
                            vertices[idx3 * 3 + 1] - vertices[idx1 * 3 + 1],
                            vertices[idx3 * 3 + 2] - vertices[idx1 * 3 + 2]);

    float3 normal1 = make_float3(v1.y * v2.z - v1.z * v2.y,
                                    v1.z * v2.x - v1.x * v2.z,
                                    v1.x * v2.y - v1.y * v2.x);

    v1 = make_float3(vertices[idx3 * 3] - vertices[idx2 * 3],
                    vertices[idx3 * 3 + 1] - vertices[idx2 * 3 + 1],
                    vertices[idx3 * 3 + 2] - vertices[idx2 * 3 + 2]);

    v2 = make_float3(vertices[idx4 * 3] - vertices[idx2 * 3],
                    vertices[idx4 * 3 + 1] - vertices[idx2 * 3 + 1],
                    vertices[idx4 * 3 + 2] - vertices[idx2 * 3 + 2]);

    float3 normal2 = make_float3(v1.y * v2.z - v1.z * v2.y,
                                v1.z * v2.x - v1.x * v2.z,
                                v1.x * v2.y - v1.y * v2.x);

    atomicAdd(&vertex_normals[idx1 * 3], normal1.x);
    atomicAdd(&vertex_normals[idx1 * 3 + 1], normal1.y);
    atomicAdd(&vertex_normals[idx1 * 3 + 2], normal1.z);
    atomicAdd(&vertex_normals[idx2 * 3], normal1.x);
    atomicAdd(&vertex_normals[idx2 * 3 + 1], normal1.y);
    atomicAdd(&vertex_normals[idx2 * 3 + 2], normal1.z);
    atomicAdd(&vertex_normals[idx3 * 3], normal1.x);
    atomicAdd(&vertex_normals[idx3 * 3 + 1], normal1.y);
    atomicAdd(&vertex_normals[idx3 * 3 + 2], normal1.z);

    atomicAdd(&vertex_normals[idx2 * 3], normal2.x);
    atomicAdd(&vertex_normals[idx2 * 3 + 1], normal2.y);
    atomicAdd(&vertex_normals[idx2 * 3 + 2], normal2.z);
    atomicAdd(&vertex_normals[idx3 * 3], normal2.x);
    atomicAdd(&vertex_normals[idx3 * 3 + 1], normal2.y);
    atomicAdd(&vertex_normals[idx3 * 3 + 2], normal2.z);
    atomicAdd(&vertex_normals[idx4 * 3], normal2.x);
    atomicAdd(&vertex_normals[idx4 * 3 + 1], normal2.y);
    atomicAdd(&vertex_normals[idx4 * 3 + 2], normal2.z);
}
'''

# 编译自定义CUDA核函数
kernel_compute_normals_quad = cp.RawKernel(kernel_code_compute_normals_quad, 'compute_normals_quad')


def cuda_compute_normals_quad(vertices, indices):

    # 将数据转移到GPU上
    vertices_gpu = cp.asarray(vertices, dtype=cp.float32)
    indices_gpu = cp.asarray(indices)

    # 顶点数量和三角形数量
    n = vertices_gpu.shape[0]
    m = indices_gpu.shape[0]

    vertex_normals_gpu = cp.zeros((n, 3), dtype=cp.float32)

    # 执行自定义核函数
    threads_per_block = 64
    blocks_per_grid = (m + threads_per_block - 1) // threads_per_block
    kernel_compute_normals_quad((blocks_per_grid,), (threads_per_block,),
                    (indices_gpu, vertices_gpu, vertex_normals_gpu, m))

    # 归一化顶点法线(np)
    vertex_normals = cp.asnumpy(vertex_normals_gpu)
    norm_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True) 
    norm_len[norm_len < 1e-5] = 1
    vertex_normals = vertex_normals / norm_len
    
    return vertex_normals


# ====================
# 插值 
# ====================
kernel_code_interp = '''
extern "C" __global__
void interp(float* mapxy, float* marker_grids, float* x_list, float* y_list, 
            float x_interval, float y_interval, int col_n, int row_n, float x_min, float y_min, int width, int height, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int yi = int(i / width);
    int xi = int(i % width);
    int2 index = make_int2(xi , yi);

    int x_idx = int((index.x - x_min) / x_interval);
    int y_idx = int((index.y - y_min) / y_interval);

    x_idx = min(max(x_idx, 0), col_n - 2);
    y_idx = min(max(y_idx, 0), row_n - 2);
   
    float2 P11 = make_float2(marker_grids[(y_idx * col_n + x_idx) * 2 + 1],
                            marker_grids[(y_idx * col_n + x_idx) * 2]);

    float2 P12 = make_float2(marker_grids[(y_idx * col_n + x_idx + 1) * 2 + 1],
                            marker_grids[(y_idx * col_n + x_idx + 1) * 2]);

    float2 P21 = make_float2(marker_grids[((y_idx + 1) * col_n + x_idx) * 2 + 1],
                            marker_grids[((y_idx + 1) * col_n + x_idx) * 2]);

    float2 P22 = make_float2(marker_grids[((y_idx + 1) * col_n + x_idx + 1) * 2 + 1],
                            marker_grids[((y_idx + 1) * col_n + x_idx + 1) * 2]);

    float x1 = x_list[x_idx], x2 = x_list[x_idx + 1];
    float y1 = y_list[y_idx], y2 = y_list[y_idx + 1];

    float u1 = index.x - x1;
    float u2 = x2 - index.x;
    float v1 = index.y - y1;
    float v2 = y2 - index.y;
    
    float den = (x2 - x1) * (y2 - y1);
    float X_ret = (P11.x * u2 * v2 + P12.x * u1 * v2 + P21.x * u2 * v1 + P22.x * u1 * v1) / den;
    float Y_ret = (P11.y * u2 * v2 + P12.y * u1 * v2 + P21.y * u2 * v1 + P22.y * u1 * v1) / den;

    //mapxy[i * 2] = Y_ret;
    //mapxy[i * 2 + 1] = X_ret;

    atomicExch(&mapxy[i * 2], Y_ret);
    atomicExch(&mapxy[i * 2 + 1], X_ret);

    }
'''

kernel_interp = cp.RawKernel(kernel_code_interp, 'interp')

def cuda_interp(
    mapxy,
    marker_grids,
    x_list,
    y_list,
    x_interval,
    y_interval,
):

    # 将数据转移到GPU上
    mapxy_gpu = cp.array(mapxy, dtype=cp.float32)
    marker_gpu = cp.array(marker_grids, dtype=cp.float32)
    x_list_gpu = cp.array(x_list, dtype=cp.float32)
    y_list_gpu = cp.array(y_list, dtype=cp.float32)
    x_interval_gpu = cp.float32(x_interval)
    y_interval_gpu = cp.float32(y_interval)

    height = mapxy.shape[0]
    width = mapxy.shape[1]
    n = height * width
    #m = marker_grids.shape[0]

    col_n = x_list.shape[0]
    row_n = y_list.shape[0]
    x_min = x_list[0]
    y_min = y_list[0]

    # 执行自定义核函数
    threads_per_block = 64
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    kernel_interp((blocks_per_grid,), (threads_per_block,),
                    (mapxy_gpu, marker_gpu, x_list_gpu, y_list_gpu, x_interval_gpu, y_interval_gpu, col_n, row_n, x_min, y_min, width, height, n))
    
    mapxy[:] = cp.asnumpy(mapxy_gpu)
