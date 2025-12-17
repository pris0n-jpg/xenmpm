"""
Description: 读入 fem 文件, 压缩, 补充信息

Author: Jin Liu
Date: 2025/01/08
"""


import numpy as np
from xengym import ASSET_DIR


if __name__ == '__main__':
    fem_file = ASSET_DIR / "data/deprecated/fem_data_vec4070.npz"
    save_file = ASSET_DIR / "data/fem_data_vec4070.npz"
    data = np.load(str(fem_file))

    # 转成 dict
    data = dict(data)
    for k, v in data.items():
        if v.dtype == np.float64:
            data[k] = v.astype(np.float32)
        print(k, v.shape, v.dtype)

    data["mesh_shape"] = np.array([70, 40], np.int32)

    # 保存
    np.savez_compressed(str(save_file), **data)