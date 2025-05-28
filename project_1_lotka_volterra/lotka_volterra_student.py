#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：李欣欣
学号：20221180076
完成日期：2025.5.28
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, 
                          gamma: float, delta: float) -> np.ndarray:
    """
    Lotka-Volterra方程组的右端函数
    
    方程组：
    dx/dt = α*x - β*x*y  (猎物增长率 - 被捕食率)
    dy/dt = γ*x*y - δ*y  (捕食者增长率 - 死亡率)
    """
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法（2阶Runge-Kutta法）求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + k2) / 2
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1/2, t[i] + dt/2, *args)
        k3 = dt * f(y[i] + k2/2, t[i] + dt/2, *args)
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], 
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解Lotka-Volterra方程组
    """
    y0_vec = np.array([x0, y0])
    t, y = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x, y = y[:, 0], y[:, 1]
    return t, x, y


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], 
                   dt: float) -> dict:
    """
    比较三种数值方法求解Lotka-Volterra方程组
    """
    y0_vec = np.array([x0, y0])
    
    # 使用欧拉法求解
    t_euler, y_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_euler, y_euler = y_euler[:, 0], y_euler[:, 1]
    
    # 使用改进欧拉法求解
    t_ie, y_ie = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_ie, y_ie = y_ie[:, 0], y_ie[:, 1]
    
    # 使用4阶龙格-库塔法求解
    t_rk4, y_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x_rk4, y_rk4 = y_rk4[:, 0], y_rk4[:, 1]
    
    results = {
        'euler': {'t': t_euler, 'x': x_euler, 'y': y_euler},
        'improved_euler': {'t': t_ie, 'x': x_ie, 'y': y_ie},
        'rk4': {'t': t_rk4, 'x': x_rk4, 'y': y_rk4}
    }
    
    return results


def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           title: str = "Lotka-Volterra种群动力学") -> None:
    """
    绘制种群动力学图
    """
    plt.figure(figsize=(12, 5))
    
    # 子图1：时间序列图
    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='猎物 (x)', color='blue')
    plt.plot(t, y, label='捕食者 (y)', color='red')
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('种群数量随时间变化')
    plt.legend()
    plt.grid(True)
    
    # 子图2：相空间轨迹图
    plt.subplot(1, 2, 2)
    plt.plot(x, y, color='green')
    plt.xlabel('猎物数量 (x)')
    plt.ylabel('捕食者数量 (y)')
    plt.title('相空间轨迹')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_method_comparison(results: dict) -> None:
    """
    绘制不同数值方法的比较图
    """
    plt.figure(figsize=(15, 10))
    
    # 上排：三种方法的时间序列图
    plt.subplot(2, 3, 1)
    plt.plot(results['euler']['t'], results['euler']['x'], label='猎物 (x)', color='blue')
    plt.plot(results['euler']['t'], results['euler']['y'], label='捕食者 (y)', color='red')
    plt.title('欧拉法')
    plt.ylabel('种群数量')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(results['improved_euler']['t'], results['improved_euler']['x'], label='猎物 (x)', color='blue')
    plt.plot(results['improved_euler']['t'], results['improved_euler']['y'], label='捕食者 (y)', color='red')
    plt.title('改进欧拉法')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(results['rk4']['t'], results['rk4']['x'], label='猎物 (x)', color='blue')
    plt.plot(results['rk4']['t'], results['rk4']['y'], label='捕食者 (y)', color='red')
    plt.title('4阶龙格-库塔法')
    plt.legend()
    plt.grid(True)
    
    # 下排：三种方法的相空间图
    plt.subplot(2, 3, 4)
    plt.plot(results['euler']['x'], results['euler']['y'], color='green')
    plt.xlabel('猎物数量')
    plt.ylabel('捕食者数量')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(results['improved_euler']['x'], results['improved_euler']['y'], color='green')
    plt.xlabel('猎物数量')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(results['rk4']['x'], results['rk4']['y'], color='green')
    plt.xlabel('猎物数量')
    plt.grid(True)
    
    plt.suptitle('不同数值方法比较')
    plt.tight_layout()
    plt.show()


def analyze_parameters() -> None:
    """
    分析不同参数对系统行为的影响
    """
    # 基本参数
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    # 1. 不同初始条件的影响
    initial_conditions = [
        (2.0, 2.0),  # 基准情况
        (1.0, 1.0),  # 低初始种群
        (4.0, 4.0),  # 高初始种群
        (2.0, 1.0),  # 捕食者较少
        (1.0, 2.0)   # 猎物较少
    ]
    
    plt.figure(figsize=(15, 8))
    for i, (x0, y0) in enumerate(initial_conditions):
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        
        plt.subplot(2, 3, i+1)
        plt.plot(t, x, label='猎物', color='blue')
        plt.plot(t, y, label='捕食者', color='red')
        plt.title(f'初始条件: x0={x0}, y0={y0}')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle('不同初始条件对种群动力学的影响')
    plt.tight_layout()
    plt.show()
    
    # 2. 守恒量验证
    x0, y0 = 2.0, 2.0
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    
    # 计算守恒量 V = γx + βy - δlnx - αlny
    V = gamma * x + beta * y - delta * np.log(x) - alpha * np.log(y)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, V, label='守恒量 V', color='purple')
    plt.axhline(y=V[0], color='red', linestyle='--', label='初始值')
    plt.title('守恒量随时间变化')
    plt.xlabel('时间')
    plt.ylabel('V = γx + βy - δlnx - αlny')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    主函数：演示Lotka-Volterra模型的完整分析
    """
    # 参数设置（根据题目要求）
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    # 1. 基本求解
    print("\n1. 使用4阶龙格-库塔法求解...")
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_population_dynamics(t, x, y)
    
    # 2. 方法比较
    print("\n2. 比较不同数值方法...")
    results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_method_comparison(results)
    
    # 3. 参数分析
    print("\n3. 分析参数影响...")
    analyze_parameters()
    
    # 4. 数值结果统计
    print("\n4. 数值结果统计:")
    print(f"猎物数量范围: {np.min(x):.2f} - {np.max(x):.2f}")
    print(f"捕食者数量范围: {np.min(y):.2f} - {np.max(y):.2f}")
    print(f"猎物周期: {t[np.argmax(np.diff(np.sign(x - np.mean(x))))]:.2f}")
    print(f"捕食者周期: {t[np.argmax(np.diff(np.sign(y - np.mean(y))))]:.2f}")


if __name__ == "__main__":
    main()
