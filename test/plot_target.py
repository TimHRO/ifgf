import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('./test/target_points.csv', delimiter=',', skiprows=1)
x, y, z = data[:, 0], data[:, 1], data[:, 2]

print(f"Loaded {len(x)} points inside the cone")

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')


# Scatter plot in 3D
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.6, marker='o')

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Z Value', fontsize=12)

# Set labels
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# Set title
ax.set_title(f'3D Visualization of Target Points Inside Cone (N = {len(x)})', 
             fontsize=14, pad=20)

# Set equal aspect ratio for all axes
max_range = max([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
]) / 2.0

mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Optional: Add a grid for better visualization
ax.grid(True, alpha=0.3)

# Adjust viewing angle for better perspective
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()