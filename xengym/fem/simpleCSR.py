import os
import numpy as np
# from ezgl.experimental import CUDA_AVAILABLE
CUDA_AVAILABLE = False
CUDA_AVAILABLE = CUDA_AVAILABLE and os.environ.get("CUDA_CSR", True) # 关闭 CUDA 加速

if CUDA_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg
    from cupyx.scipy.sparse import csr_matrix
else:
    from scipy.sparse.linalg import cg
    from scipy.sparse import csr_matrix


class SimpleCSR:

    def __init__(self, data, indices, indptr, shape, top_diag_offset_in_data=None):
        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = shape

        # 实际的 csr_matrix 对象
        self._csr_matrix = None
        self._csr_matrix_pending = True

        if top_diag_offset_in_data is None:
            # 把顶面节点 xyz 对应的对角元都添加上值用于在 data 中占位, 避免频繁 insert, 顶面节点是前 1/3 个节点
            self.top_diag_offset_in_data = np.zeros(self.shape[0]//3, dtype=np.int32)
            for i in range(self.shape[0]//3):
                offset = self._set((i, i), self[i, i])
                self.top_diag_offset_in_data[i] = offset
        else:
            self.top_diag_offset_in_data = top_diag_offset_in_data  # no need to copy

    @property
    def csr_matrix(self):
        if self._csr_matrix_pending:
            self._csr_matrix = self._to_csr_matrix()
            self._csr_matrix_pending = False
        return self._csr_matrix

    def set_row_zero(self, start_row, end_row):
        """
        把矩阵的连续行设为 0
        """
        self._csr_matrix_pending = True
        self.data[self.indptr[start_row]:self.indptr[end_row]] = 0

    def set_diag_one(self, idx):
        """
        把矩阵的对角元设为 1
        """
        self._csr_matrix_pending = True
        offset = self.top_diag_offset_in_data[idx]
        self.data[offset] = 1

    def __getitem__(self, idx):
        """获取矩阵元素，通过 (row, col) 索引"""
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            # 查找对应的列
            start, end = self.indptr[row], self.indptr[row + 1]
            for i in range(start, end):
                if self.indices[i] == col:
                    return self.data[i]
            return 0  # 若未找到，返回0
        else:
            raise NotImplementedError("仅支持 (row, col) 索引方式")

    def _set(self, idx, value):
        """通过 (row, col) 索引设置矩阵元素, 返回元素在 data 中的索引"""
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            start, end = self.indptr[row], self.indptr[row + 1]
            for i in range(start, end):
                if self.indices[i] == col:
                    self.data[i] = value
                    return i
            # 若没有找到该位置的元素，则需要在数据中插入
            self.indices = np.insert(self.indices, end, col)
            self.data = np.insert(self.data, end, value)
            self.indptr[row + 1:] += 1  # 更新后续行的指针
        else:
            raise NotImplementedError("仅支持 (row, col) 索引方式")
        return end

    def copy(self):
        return SimpleCSR(self.data, self.indices, self.indptr, self.shape, self.top_diag_offset_in_data)

    def _to_csr_matrix(self):
        if CUDA_AVAILABLE:
            return csr_matrix((cp.array(self.data), cp.array(self.indices), cp.array(self.indptr)), shape=tuple(self.shape))
        else:
            return csr_matrix((self.data, self.indices, self.indptr), shape=tuple(self.shape))

    def __matmul__(self, other):
        if CUDA_AVAILABLE:
            return (self.csr_matrix @ other).get()
        else:
            return self.csr_matrix @ other

    @classmethod
    def solve(cls, A: 'SimpleCSR', P: np.ndarray, x0: np.ndarray):
        """
        使用共轭梯度法求解线性方程组 Ax = P

        Parameters:
        - A : SimpleCSR, 系数矩阵
        - P : array-like,
        - x0 : array-like, 初始估计

        Returns:
        - x : array-like, 位移
        """
        if CUDA_AVAILABLE:
            return cg(A.csr_matrix, cp.array(P), x0=cp.array(x0), tol=0.01, maxiter=90)
        else:
            return cg(A.csr_matrix, P, x0=x0, rtol=0.02, maxiter=50)