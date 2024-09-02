import numpy as np

position = np.array([1.0, 2.0, 3.0])
# 创建 3x3 的旋转矩阵
rotation_matrix = np.array([[-0.11389835, -0.00884288, -0.99345305],
                           [0.95773053, -0.26685497, -0.10742749],
                           [-0.26415792, -0.96369613, 0.03886343]])

# 创建 4x4 的齐次变换矩阵
transform_matrix = np.eye(4, dtype=np.float32)
transform_matrix[:3, :3] = rotation_matrix
transform_matrix[:3, 3] = position
print(transform_matrix)