import numpy as np
import matplotlib.pyplot as plt

# === 1. Генерація даних ===
TRUE_K = 2
TRUE_B = 1
NUM_POINTS = 100

x_vals = np.linspace(-10, 10, NUM_POINTS)
noise = np.random.normal(loc=0.0, scale=2.0, size=NUM_POINTS)
y_vals = TRUE_K * x_vals + TRUE_B + noise
x_sorted = np.sort(x_vals)
true_line = TRUE_K * x_sorted + TRUE_B

# === 2. МНК вручну ===
def manual_linear_regression(x, y):
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    slope = np.sum((x - x_avg) * (y - y_avg)) / np.sum((x - x_avg) ** 2)
    intercept = y_avg - slope * x_avg

    return slope, intercept

slope_manual, intercept_manual = manual_linear_regression(x_vals, y_vals)
y_manual = slope_manual * x_sorted + intercept_manual

# === 3. МНК через numpy.polyfit ===
coeffs = np.polyfit(x_vals, y_vals, deg=1)
slope_np, intercept_np = coeffs
y_numpy = slope_np * x_sorted + intercept_np

# === 4. Градієнтний спуск ===
def linear_gradient_descent(x, y, lr=0.01, steps=1000):
    slope = 0.0
    intercept = 0.0
    m = len(x)
    errors = []

    for _ in range(steps):
        pred = slope * x + intercept
        err = pred - y

        mse = np.mean(err ** 2)
        errors.append(mse)

        grad_slope = (2/m) * np.dot(err, x)
        grad_intercept = (2/m) * np.sum(err)

        slope -= lr * grad_slope
        intercept -= lr * grad_intercept

    return slope, intercept, errors

ITERATIONS = 10
slope_gd, intercept_gd, loss_history = linear_gradient_descent(x_vals, y_vals, lr=0.01, steps=ITERATIONS)
y_gd = slope_gd * x_sorted + intercept_gd

# === 5. Візуалізація ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Графік регресій
ax1.scatter(x_vals, y_vals, label='Шумні дані', alpha=0.6)
ax1.plot(x_sorted, true_line, 'r', label=f'Істинна лінія: y = {TRUE_K}x + {TRUE_B}')
ax1.plot(x_sorted, y_manual, 'b', label=f'МНК вручну: y = {slope_manual:.2f}x + {intercept_manual:.2f}')
ax1.plot(x_sorted, y_numpy, 'g--', label=f'numpy.polyfit: y = {slope_np:.2f}x + {intercept_np:.2f}')
ax1.plot(x_sorted, y_gd, 'k--', label=f'Градієнт: y = {slope_gd:.2f}x + {intercept_gd:.2f}')
ax1.set_title('Лінії регресії')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# Графік помилки
ax2.plot(np.arange(ITERATIONS), loss_history, color='darkorange')
ax2.set_title('Збіжність градієнтного спуску')
ax2.set_xlabel('Ітерації')
ax2.set_ylabel('Середньоквадратична помилка (MSE)')
ax2.grid(True)

plt.tight_layout()
plt.show()
