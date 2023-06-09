import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8  # Acceleration due to gravity (m/s^2)

# Video frame dimensions
frame_width = 1280
frame_height = 720

# Set initial position
initial_x = 710 # Center of the frame
initial_y = 418 // 2  # Center of the frame
pos_initial = np.array([initial_x, initial_y])

# Set initial velocity
initial_speed = 10  # Units per time step (e.g., pixels per frame)
initial_angle = 45  # Angle in degrees (e.g., 45 degrees from the horizontal axis)
angle_rad = np.radians(initial_angle)
vx = initial_speed * np.cos(angle_rad)
vy = -initial_speed * np.sin(angle_rad)  # Negative sign to account for inverted y-axis
vel_initial = np.array([vx, vy])

# Initial parameters
dt = 0.1  # Time step size (s)
t_total = 20.0  # Total simulation time (s)
# pos_initial = np.array([710, 418])  # Initial position (m)
# vel_initial = np.array([10, 20])  # Initial velocity (m/s)

# Calculate the number of time steps
num_steps = int(t_total / dt)

# Arrays to store the position and velocity at each time step
pos = np.zeros((num_steps, 2))
vel = np.zeros((num_steps, 2))

# Set the initial position and velocity
pos[0] = pos_initial
vel[0] = vel_initial

# Simulate the motion of the ball
for i in range(1, num_steps):
    # Update the position and velocity using the dynamics equations
    pos[i] = pos[i-1] + vel[i-1] * dt
    vel[i] = vel[i-1] + np.array([0, -g]) * dt

# Plot the trajectory of the ball
plt.plot(pos[:, 0], pos[:, 1])
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Ball Trajectory')
plt.grid(True)
plt.show()
