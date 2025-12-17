import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from identifier.visualization import Animation, visualize_mesh, visualize_fem_result
from identifier.utils import convert_excel_to_numpy, convert_csv_to_numpy

DATA_DIR = os.path.join(project_root, 'collector', 'data', 'sensor_calibration', 'sensor_0_soft_new_x')

def main():
    # read all .csv files in the folder
    all_data = []
    file_name_list = []
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith('.csv'):
            file_name_list.append(file_name)
            file_path = os.path.join(DATA_DIR, file_name)
            data = convert_csv_to_numpy(file_path)
            if data is not None:
                all_data.append(data)

    # data names of each column
    column_names = {
        'id': 0,
        'x': 1,
        'y': 2,
        'z': 3,
        'fx': 4,
        'fy': 5,
        'fz': 6,
        'indent': 7,
        'contact_node_num': 8,
        'ux': 9,
        'uy': 10,
        'uz': 11,
        'ux_square': 12,
        'uy_square': 13,
        'uz_square': 14
    }

    print(f'Total {len(all_data)} data files loaded.')

    print('Fit displacement data pairs and visualize fitting results...')

    # visualize the data: plot curves of specific columns of all_data
    # sensor_0_soft_new_y corresponds to ux in sensor local frame
    target_x = 'uy'
    # target_x = 'ux'
    target_y = 'indent'
    x_column = column_names[target_x]
    y_column = column_names[target_y]
    plt.figure()
    for data in all_data:
        plt.plot(np.abs(data[1:, x_column]), np.abs(data[1:, y_column]), label=f'File {data[0,0]}', marker='o')
    plt.xlabel(x_column)
    plt.ylabel(target_y)
    plt.title('displacement data pairs for all data files')
    plt.legend(file_name_list)
    plt.show()

    # fit a curve for total data points and print the fitting parameters (linear & quadratic), and intercept is zero, and calculate R^2
    # remove first data row
    all_x = np.abs(np.concatenate([data[1:, x_column] for data in all_data]))
    all_y = np.abs(np.concatenate([data[1:, y_column] for data in all_data]))

    # 准备设计矩阵
    # 标准线性拟合（有截距）
    A_linear = np.vstack([all_x, np.ones(len(all_x))]).T
    linear_fit, _, _, _ = np.linalg.lstsq(A_linear, all_y, rcond=None)
    slope_linear, intercept_linear = linear_fit
    print(f'Linear fit parameters (slope, intercept): [{slope_linear}, {intercept_linear}]')

    # 强制截距为零的线性拟合
    A_zero = all_x.reshape(-1, 1)
    zero_fit, _, _, _ = np.linalg.lstsq(A_zero, all_y, rcond=None)
    zero_intercept_slope = zero_fit[0]
    print(f'Zero-intercept linear fit (slope): {zero_intercept_slope}')

    # 标准二次拟合
    A_quad = np.vstack([all_x**2, all_x, np.ones(len(all_x))]).T
    quad_fit, _, _, _ = np.linalg.lstsq(A_quad, all_y, rcond=None)
    a_quad, b_quad, c_quad = quad_fit
    print(f'Quadratic fit parameters (a, b, c): [{a_quad}, {b_quad}, {c_quad}]')

    # 强制截距为零的二次拟合
    A_quad_zero = np.vstack([all_x**2, all_x]).T
    quad_zero_fit, _, _, _ = np.linalg.lstsq(A_quad_zero, all_y, rcond=None)
    a_quad_zero, b_quad_zero = quad_zero_fit
    print(f'Zero-intercept quadratic fit (a, b): [{a_quad_zero}, {b_quad_zero}]')

    # 计算预测值
    y_linear_pred = slope_linear * all_x + intercept_linear
    y_zero_pred = zero_intercept_slope * all_x
    y_quad_pred = a_quad * all_x**2 + b_quad * all_x + c_quad
    y_quad_zero_pred = a_quad_zero * all_x**2 + b_quad_zero * all_x

    # 计算平均值
    y_mean = np.mean(all_y)

    # 计算总平方和 (TSS)
    ss_total = np.sum((all_y - y_mean) ** 2)

    # 计算残差平方和 (RSS)
    ss_res_linear = np.sum((all_y - y_linear_pred) ** 2)
    ss_res_zero = np.sum((all_y - y_zero_pred) ** 2)
    ss_res_quad = np.sum((all_y - y_quad_pred) ** 2)
    ss_res_quad_zero = np.sum((all_y - y_quad_zero_pred) ** 2)

    # 计算R²
    r2_linear = 1 - (ss_res_linear / ss_total)
    r2_zero = 1 - (ss_res_zero / ss_total)
    r2_quad = 1 - (ss_res_quad / ss_total)
    r2_quad_zero = 1 - (ss_res_quad_zero / ss_total)

    print(f'R² for linear fit: {r2_linear:.4f}')
    print(f'R² for zero-intercept linear fit: {r2_zero:.4f}')
    print(f'R² for quadratic fit: {r2_quad:.4f}')
    print(f'R² for zero-intercept quadratic fit: {r2_quad_zero:.4f}')

    # plot the fitting curves and original data
    plt.figure(figsize=(10, 8))
    plt.scatter(all_x, all_y, color='gray', alpha=0.3, label='Data points')

    # 生成拟合曲线的点
    x_fit = np.linspace(min(all_x), max(all_x), 100)
    y_linear_fit = slope_linear * x_fit + intercept_linear
    y_zero_fit = zero_intercept_slope * x_fit
    y_quad_fit = a_quad * x_fit**2 + b_quad * x_fit + c_quad
    y_quad_zero_fit = a_quad_zero * x_fit**2 + b_quad_zero * x_fit

    # 绘制拟合曲线
    plt.plot(x_fit, y_linear_fit, color='blue', label=f'Linear (R²={r2_linear:.4f})', linewidth=2, alpha=0.3)
    plt.plot(x_fit, y_zero_fit, color='green', label=f'Zero-intercept linear (R²={r2_zero:.4f})', linewidth=2)
    plt.plot(x_fit, y_quad_fit, color='red', label=f'Quadratic (R²={r2_quad:.4f})', linewidth=2, alpha=0.3)
    plt.plot(x_fit, y_quad_zero_fit, color='purple', label=f'Zero-intercept quadratic (R²={r2_quad_zero:.4f})', linewidth=2)

    plt.xlabel(target_x)
    plt.ylabel(target_y)
    plt.title('Curve Fitting for displacement data pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    print('Fit force-displacement relationship and visualize fitting results...')
    def resultant_force(fx, fy):
        return np.sqrt(fx**2 + fy**2)

    # visualize the data: plot curves of specific columns of all_data
    target_x = 'indent'
    target_y1 = 'fx'
    target_y2 = 'fy'
    x_column = column_names[target_x]
    y1_column = column_names[target_y1]
    y2_column = column_names[target_y2]
    plt.figure()
    for data in all_data:
        plt.plot(np.abs(data[1:, x_column]), resultant_force(data[1:, y1_column], data[1:, y2_column]), label=f'File {data[0,0]}', marker='o')
    plt.xlabel(x_column)
    plt.ylabel(target_y)
    plt.title('force-displacement relationship for all data files')
    plt.legend(file_name_list)
    plt.figure()
    for data in all_data:
        plt.plot(np.abs(data[1:, x_column]), resultant_force(data[1:, y1_column], data[1:, y2_column])/data[1:,column_names['contact_node_num']], label=f'File {data[0,0]}', marker='o')
    plt.xlabel(x_column)
    plt.ylabel(target_y)
    plt.title('nodal force-displacement relationship for all data files')
    plt.legend(file_name_list)
    plt.show()

    # fit a curve for total data points and print the fitting parameters (linear & quadratic), and intercept is zero, and calculate R^2
    all_x = np.abs(np.concatenate([data[1:, x_column] for data in all_data]))
    # all_y = np.abs(np.concatenate([resultant_force(data[1:, y1_column], data[1:, y2_column]) for data in all_data]))
    all_average_node_force = np.abs(np.concatenate([resultant_force(data[1:, y1_column], data[1:, y2_column])/data[1:, column_names['contact_node_num']] for data in all_data]))
    all_y = all_average_node_force

    # 准备设计矩阵
    # 标准线性拟合（有截距）
    A_linear = np.vstack([all_x, np.ones(len(all_x))]).T
    linear_fit, _, _, _ = np.linalg.lstsq(A_linear, all_y, rcond=None)
    slope_linear, intercept_linear = linear_fit
    print(f'Linear fit parameters (slope, intercept): [{slope_linear}, {intercept_linear}]')

    # 强制截距为零的线性拟合
    A_zero = all_x.reshape(-1, 1)
    zero_fit, _, _, _ = np.linalg.lstsq(A_zero, all_y, rcond=None)
    zero_intercept_slope = zero_fit[0]
    print(f'Zero-intercept linear fit (slope): {zero_intercept_slope}')

    # 标准二次拟合
    A_quad = np.vstack([all_x**2, all_x, np.ones(len(all_x))]).T
    quad_fit, _, _, _ = np.linalg.lstsq(A_quad, all_y, rcond=None)
    a_quad, b_quad, c_quad = quad_fit
    print(f'Quadratic fit parameters (a, b, c): [{a_quad}, {b_quad}, {c_quad}]')

    # 强制截距为零的二次拟合
    A_quad_zero = np.vstack([all_x**2, all_x]).T
    quad_zero_fit, _, _, _ = np.linalg.lstsq(A_quad_zero, all_y, rcond=None)
    a_quad_zero, b_quad_zero = quad_zero_fit
    print(f'Zero-intercept quadratic fit (a, b): [{a_quad_zero}, {b_quad_zero}]')

    # 计算预测值
    y_linear_pred = slope_linear * all_x + intercept_linear
    y_zero_pred = zero_intercept_slope * all_x
    y_quad_pred = a_quad * all_x**2 + b_quad * all_x + c_quad
    y_quad_zero_pred = a_quad_zero * all_x**2 + b_quad_zero * all_x

    # 计算平均值
    y_mean = np.mean(all_y)

    # 计算总平方和 (TSS)
    ss_total = np.sum((all_y - y_mean) ** 2)

    # 计算残差平方和 (RSS)
    ss_res_linear = np.sum((all_y - y_linear_pred) ** 2)
    ss_res_zero = np.sum((all_y - y_zero_pred) ** 2)
    ss_res_quad = np.sum((all_y - y_quad_pred) ** 2)
    ss_res_quad_zero = np.sum((all_y - y_quad_zero_pred) ** 2)

    # 计算R²
    r2_linear = 1 - (ss_res_linear / ss_total)
    r2_zero = 1 - (ss_res_zero / ss_total)
    r2_quad = 1 - (ss_res_quad / ss_total)
    r2_quad_zero = 1 - (ss_res_quad_zero / ss_total)

    print(f'R² for linear fit: {r2_linear:.4f}')
    print(f'R² for zero-intercept linear fit: {r2_zero:.4f}')
    print(f'R² for quadratic fit: {r2_quad:.4f}')
    print(f'R² for zero-intercept quadratic fit: {r2_quad_zero:.4f}')

    # plot the fitting curves and original data
    plt.figure(figsize=(10, 8))
    plt.scatter(all_x, all_y, color='gray', alpha=0.3, label='Data points')

    # 生成拟合曲线的点
    x_fit = np.linspace(min(all_x), max(all_x), 100)
    y_linear_fit = slope_linear * x_fit + intercept_linear
    y_zero_fit = zero_intercept_slope * x_fit
    y_quad_fit = a_quad * x_fit**2 + b_quad * x_fit + c_quad
    y_quad_zero_fit = a_quad_zero * x_fit**2 + b_quad_zero * x_fit

    # 绘制拟合曲线
    plt.plot(x_fit, y_linear_fit, color='blue', label=f'Linear (R²={r2_linear:.4f})', linewidth=2, alpha=0.3)
    plt.plot(x_fit, y_zero_fit, color='green', label=f'Zero-intercept linear (R²={r2_zero:.4f})', linewidth=2)
    plt.plot(x_fit, y_quad_fit, color='red', label=f'Quadratic (R²={r2_quad:.4f})', linewidth=2, alpha=0.3)
    plt.plot(x_fit, y_quad_zero_fit, color='purple', label=f'Zero-intercept quadratic (R²={r2_quad_zero:.4f})', linewidth=2)

    plt.xlabel(target_x)
    plt.ylabel(target_y)
    plt.title('Curve Fitting for nodal force-displacement relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()








if __name__ == '__main__':
    main()
