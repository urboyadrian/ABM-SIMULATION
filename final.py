import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
import os
import time
import subprocess

# GUI Root
root = tk.Tk()
root.title("Epidemic Model Simulation")
root.geometry("400x500")

# Model Selection
ttk.Label(root, text="Select Model:").pack()
model_var = tk.StringVar(value="SIR")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["SIR", "SEIR", "SIRS", "SEIRS"])
model_dropdown.pack()

# Input Fields
entry_vars = {}
params = [
    ("Population Size (N):", "500"),
    ("Grid Size:", "50"),
    ("Infection Radius:", "2"),
    ("Transmission Rate (β):", "0.7"),
    ("Recovery Rate (γ):", "0.03"),
    ("Incubation Rate (σ):", "0.2"),   # For SEIR models
    ("Susceptibility Rate (δ):", "0.01"),  # For SIRS/SEIRS models
    ("Time Steps (Days):", "100")
]

for label, default in params:
    ttk.Label(root, text=label).pack()
    entry_var = tk.StringVar(value=default)
    entry = ttk.Entry(root, textvariable=entry_var)
    entry.pack()
    entry_vars[label] = entry_var  # Store variables for later use


def run_simulation():
    """Runs the chosen epidemiological model and generates two animations."""
    # Get parameters from input fields
    N = int(entry_vars["Population Size (N):"].get())
    grid_size = int(entry_vars["Grid Size:"].get())
    infection_radius = int(entry_vars["Infection Radius:"].get())
    beta = float(entry_vars["Transmission Rate (β):"].get())
    gamma = float(entry_vars["Recovery Rate (γ):"].get())
    sigma = float(entry_vars["Incubation Rate (σ):"].get()) if "SEIR" in model_var.get() else 0
    delta = float(entry_vars["Susceptibility Rate (δ):"].get()) if "SIRS" in model_var.get() else 0
    time_steps = int(entry_vars["Time Steps (Days):"].get())
    model = model_var.get()

    # Generate filenames
    timestamp = int(time.time())
    agent_gif = f"disease_spread_simulation_{timestamp}.gif"
    sir_gif = f"SIR_spread_simulation_{timestamp}.gif"

    # Initialize agents & states
    agents = np.random.randint(0, grid_size, (N, 2))
    states = np.zeros(N, dtype=int)  # 0: S, 1: I, 2: E (for SEIR), 3: R
    states[np.random.choice(N, size=5, replace=False)] = 1  # Start with infected

    # Track counts
    S_counts, E_counts, I_counts, R_counts = [], [], [], []

    # Plot setup
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(0, grid_size)
    ax1.set_title("Agent-Based Simulation")

    def update_agents(frame):
        """Updates agent positions and infection states."""
        nonlocal agents, states

        ax1.clear()
        ax1.set_xlim(0, grid_size)
        ax1.set_ylim(0, grid_size)
        ax1.set_title(f"Disease Spread Simulation (Day {frame})")

        # Move agents randomly
        agents += np.random.randint(-1, 2, (N, 2))
        agents = np.clip(agents, 0, grid_size - 1)

        # Infection Spread
        for i in range(N):
            if states[i] in [1, 2]:  # Infected or Exposed (for SEIR)
                for j in range(N):
                    if states[j] == 0 and np.linalg.norm(agents[i] - agents[j]) < infection_radius:
                        if np.random.rand() < beta:
                            states[j] = 2 if "SEIR" in model else 1  # Exposed if SEIR

        # Disease Progression
        for i in range(N):
            if model in ["SEIR", "SEIRS"] and states[i] == 2:  # Exposed → Infected
                if np.random.rand() < sigma:
                    states[i] = 1

            if states[i] == 1 and np.random.rand() < gamma:  # Infected → Recovered
                states[i] = 3

            if model in ["SIRS", "SEIRS"] and states[i] == 3:  # Recovered → Susceptible
                if np.random.rand() < delta:
                    states[i] = 0

        # Update counts
        S_counts.append(np.sum(states == 0))
        E_counts.append(np.sum(states == 2) if "SEIR" in model else 0)
        I_counts.append(np.sum(states == 1))
        R_counts.append(np.sum(states == 3))

        # Assign colors
        color_map = {0: "blue", 1: "red", 2: "orange", 3: "green"}
        colors = np.array([color_map[state] for state in states])

        ax1.scatter(agents[:, 0], agents[:, 1], c=colors, alpha=0.6)

    ani1 = animation.FuncAnimation(fig1, update_agents, frames=time_steps, interval=200)
    ani1.save(agent_gif, writer="pillow", fps=10)
    fig1.savefig(f"agent_based_snapshot_{timestamp}.png")

    # Second Graph (Epidemiological Model)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.set_xlim(0, time_steps)
    ax2.set_ylim(0, N)
    ax2.set_title(f"{model} Model - Disease Spread Over Time")
    ax2.set_xlabel("Time Steps (Days)")
    ax2.set_ylabel("Number of Individuals")

    line_s, = ax2.plot([], [], label="Susceptible", color="blue")
    line_e, = ax2.plot([], [], label="Exposed", color="orange") if "SEIR" in model else (None, )
    line_i, = ax2.plot([], [], label="Infected", color="red")
    line_r, = ax2.plot([], [], label="Recovered", color="green")
    ax2.legend()
    ax2.grid()

    def update_chart(frame):
        """Updates the epidemiological plot dynamically."""
        if frame < len(S_counts):
            line_s.set_data(range(frame), S_counts[:frame])
            line_i.set_data(range(frame), I_counts[:frame])
            line_r.set_data(range(frame), R_counts[:frame])
            if "SEIR" in model:
                line_e.set_data(range(frame), E_counts[:frame])
        return line_s, line_i, line_r, line_e

    ani2 = animation.FuncAnimation(fig2, update_chart, frames=time_steps, interval=100)
    ani2.save(sir_gif, writer="pillow", fps=10)
    fig2.savefig(f"{model}_model_snapshot_{timestamp}.png")

    # Open GIFs side by side
    try:
        if os.name == "nt":
            os.startfile(agent_gif)
            os.startfile(sir_gif)
        else:
            subprocess.Popen(["xdg-open", agent_gif])
            subprocess.Popen(["xdg-open", sir_gif])

        time.sleep(1)

        if os.name == "nt":
            import pygetwindow as gw
            windows = [win for win in gw.getWindowsWithTitle("") if "gif" in win.title.lower()]
            if len(windows) >= 2:
                # Position for the first graph (left side of the screen)
                windows[0].moveTo(100, 600)
                # Position for the second graph (right side of the screen)
                windows[1].moveTo(700, 600)  # Adjust the X value for right position
        else:  # macOS/Linux: ensure both windows are side by side
            subprocess.Popen(["wmctrl", "-r", agent_gif, "-e", "0,100,600,-1,-1"])  # Left half of the screen
            subprocess.Popen(["wmctrl", "-r", sir_gif, "-e", "0,900,600,-1,-1"])  # Right half of the screen

    except Exception as e:
        print("Error repositioning windows:", e)


# Run Button
ttk.Button(root, text="Run Simulation", command=run_simulation).pack(pady=20)

root.mainloop()