import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

space_dim = 10  # 10 spatial points
time_dim = 52   # 52 weeks

# Generate correlated 2D spatial actual ET data
mean = np.zeros(space_dim)
covariance = np.eye(space_dim)  # Identity matrix for simplicity (can be adjusted)

# Adjust covariance matrix for correlation among spatial points
# For example, let's create a positive correlation between adjacent spatial points
# You can adjust this covariance matrix based on your desired correlation structure
for i in range(space_dim - 1):
    covariance[i, i+1] = covariance[i+1, i] = 0.5

correlated_et = np.random.multivariate_normal(mean, covariance, time_dim).T

# Scale the data to vary between 0 and 1e-7 m/s
correlated_et = correlated_et * 0.5e-7 + 0.5e-7

# Add 5% noise to the correlated ET data
noise_level = 0.05
noise = np.random.normal(0, noise_level * np.mean(correlated_et), size=correlated_et.shape)
correlated_et += noise

# Clip values to ensure they are within the specified range
correlated_et = np.clip(correlated_et, 0, 1e-7)

# Display results
print("Standard Deviations:\n", std_et)
print("Correlation Matrix:\n", correlation_matrix)
print("Covariance Matrix:\n", covariance_matrix)

# Plot the original data
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for i in range(space_dim):
    plt.plot(correlated_et[i], label=f'Spatial Point {i+1}')
plt.title('Original Data')
plt.xlabel('Weeks')
plt.ylabel('ET Measurement (m/s)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the covariance matrix using imshow
plt.figure(figsize=(10, 8))
plt.imshow(covariance_matrix, interpolation='nearest', cmap='coolwarm')
plt.colorbar()
plt.xticks(ticks=np.arange(space_dim), labels=[f'Point {i+1}' for i in range(space_dim)], rotation=45)
plt.yticks(ticks=np.arange(space_dim), labels=[f'Point {i+1}' for i in range(space_dim)])
plt.title('Covariance Matrix')
plt.show()

#%%ù

import numpy as np
import matplotlib.pyplot as plt

# Assuming `actual_et` is your 2D array of ET measurements (space_dim x time_dim)

# Choose three different time points
time_points = [10, 25, 40]

# Plot the ET measurements at the selected time points using imshow
plt.figure(figsize=(12, 6))

for i, t in enumerate(time_points, 1):
    plt.subplot(1, 3, i)
    plt.imshow(correlated_et[:, t].reshape(1, -1), cmap='viridis', aspect='auto')
    plt.colorbar(label='ET Measurement (m/s)')
    plt.title(f'ET Measurement at Time Point {t}')
    plt.xlabel('Spatial Point')
    plt.ylabel('')

plt.tight_layout()
plt.show()
