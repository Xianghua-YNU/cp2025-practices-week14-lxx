import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(t: float, state: np.ndarray, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
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
    k1 = dt * ode_func(t, state, **kwargs)
    k2 = dt * ode_func(t + dt/2, state + k1/2, **kwargs)
    k3 = dt * ode_func(t + dt/2, state + k2/2, **kwargs)
    k4 = dt * ode_func(t + dt, state + k3, **kwargs)
    return state + (k1 + 2*k2 + 2*k3 + k4)/6

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
    t = np.arange(t_start, t_end + dt, dt)
    states = np.zeros((len(t), len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, len(t)):
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

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率
    
    返回:
        float: 系统的能量
    """
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # 取后半部分数据以确保系统已达到稳态
    steady_states = states[len(states)//2:]
    x = steady_states[:, 0]
    
    # 计算振幅
    amplitude = np.max(np.abs(x))
    
    # 计算周期
    # 找到过零点的位置
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) >= 2:
        period = 2 * (zero_crossings[-1] - zero_crossings[-2])
    else:
        period = 0
    
    return amplitude, period

def plot_energy_evolution(t: np.ndarray, states: np.ndarray, omega: float, title: str) -> None:
    """
    绘制能量随时间的变化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        omega: float, 角频率
        title: str, 图标题
    """
    energy = np.array([calculate_energy(state, omega) for state in states])
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, energy)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    # 设置基本参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # 任务1 - 基本实现
    print("Task 1: Basic implementation with μ=1")
    mu = 1.0
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f"Time Evolution (μ={mu})")
    plot_phase_space(states, f"Phase Space (μ={mu})")
    plot_energy_evolution(t, states, omega, f"Energy Evolution (μ={mu})")
    
    amplitude, period = analyze_limit_cycle(states)
    print(f"For μ={mu}: Amplitude={amplitude:.2f}, Period={period*dt:.2f}")
    
    # 任务2 - 参数影响分析
    print("\nTask 2: Parameter influence analysis")
    for mu in [1.0, 2.0, 4.0]:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f"Time Evolution (μ={mu})")
        plot_phase_space(states, f"Phase Space (μ={mu})")
        
        amplitude, period = analyze_limit_cycle(states)
        print(f"For μ={mu}: Amplitude={amplitude:.2f}, Period={period*dt:.2f}")
    
    # 任务3 - 不同初始条件
    print("\nTask 3: Different initial conditions")
    mu = 2.0
    for x0, v0 in [(0.1, 0), (2.0, 0), (0, 2.0)]:
        t, states = solve_ode(van_der_pol_ode, np.array([x0, v0]), t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f"Phase Space (μ={mu}, x0={x0}, v0={v0})")

if __name__ == "__main__":
    main()
