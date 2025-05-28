import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x**3
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + 0.5*k1, t + 0.5*dt, **kwargs)
    k3 = dt * ode_func(state + 0.5*k2, t + 0.5*dt, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, num_steps)
    states = np.zeros((num_steps, len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, num_steps):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, states[:, 0], label='Displacement (x)')
    plt.plot(t, states[:, 1], label='Velocity (v)')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (v)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # 找到位移的过零点
    x = states[:, 0]
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    
    if len(zero_crossings) < 2:
        return np.nan
    
    # 计算连续过零点之间的时间差
    periods = np.diff(t[zero_crossings])
    
    # 返回平均周期
    return np.mean(periods[::2]) * 2  # 乘以2得到完整周期

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # 任务1 - 简谐振子的数值求解
    initial_state = np.array([1.0, 0.0])
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, 'Harmonic Oscillator: Time Evolution (x0=1)')
    
    # 计算并打印周期
    period = analyze_period(t, states)
    print(f"Harmonic Oscillator Period (x0=1): {period:.4f} (Theoretical: {2*np.pi/omega:.4f})")
    
    # 任务2 - 振幅对周期的影响分析
    for x0 in [0.5, 1.0, 2.0]:
        initial_state = np.array([x0, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f"Harmonic Oscillator Period (x0={x0}): {period:.4f}")
    
    # 任务3 - 非谐振子的数值分析
    initial_state = np.array([1.0, 0.0])
    t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, 'Anharmonic Oscillator: Time Evolution (x0=1)')
    
    # 分析不同振幅下的周期
    for x0 in [0.5, 1.0, 2.0]:
        initial_state = np.array([x0, 0.0])
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        print(f"Anharmonic Oscillator Period (x0={x0}): {period:.4f}")
    
    # 任务4 - 相空间分析
    # 简谐振子
    initial_state = np.array([1.0, 0.0])
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states, 'Harmonic Oscillator: Phase Space')
    
    # 非谐振子
    initial_state = np.array([1.0, 0.0])
    t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states, 'Anharmonic Oscillator: Phase Space')

if __name__ == "__main__":
    main()
