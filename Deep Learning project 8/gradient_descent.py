# ============================================================
# Gradient Descent Implementation Using Python (from scratch)
# Works in Jupyter Notebook / Google Colab
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. CREATE THE DATASET ----------
np.random.seed(42)                       # same random numbers every run
num_samples = 100
X = np.random.uniform(0, 10, num_samples)            # inputs between 0 and 10
true_weight, true_bias = 3.0, 5.0                    # the "hidden" true line
noise = np.random.normal(0, 2, num_samples)          # random noise
Y = true_weight * X + true_bias + noise              # y = 3x + 5 + noise

# ---------- 2. GRADIENT DESCENT FUNCTION ----------
def gradient_descent(X, Y, learning_rate, iterations):
    """Fits y = w*x + b by minimizing Mean Squared Error."""
    n = len(X)
    w, b = 0.0, 0.0                      # initialize parameters
    history = {"w": [], "b": [], "loss": []}

    for i in range(iterations + 1):
        y_pred = w * X + b                           # prediction
        error = y_pred - Y                           # prediction error
        loss = np.mean(error ** 2)                   # MSE loss

        history["w"].append(w)                       # store values (iteration i)
        history["b"].append(b)
        history["loss"].append(loss)

        if i == iterations:                          # stop after recording last state
            break

        dw = (2 / n) * np.sum(error * X)             # gradient wrt w
        db = (2 / n) * np.sum(error)                 # gradient wrt b

        w = w - learning_rate * dw                   # update weight
        b = b - learning_rate * db                   # update bias
        if not np.isfinite(loss) or loss > 1e12:     # stop if it blows up
            history["w"].append(w); history["b"].append(b); history["loss"].append(np.inf)
            break
    return w, b, history

# ---------- 3. MAIN TRAINING RUN (learning rate = 0.01) ----------
learning_rate, iterations = 0.01, 1000
w, b, hist = gradient_descent(X, Y, learning_rate, iterations)

print("Training table (learning rate = 0.01)")
print(f"{'Iteration':>10} {'Weight':>10} {'Bias':>10} {'Loss':>12}")
for it in [0, 100, 200, 500]:
    print(f"{it:>10} {hist['w'][it]:>10.4f} {hist['b'][it]:>10.4f} {hist['loss'][it]:>12.4f}")
print(f"{'Final':>10} {w:>10.4f} {b:>10.4f} {hist['loss'][-1]:>12.4f}")
print(f"\nTrue values used to create data: w = {true_weight}, b = {true_bias}")

# ---------- 4. GRAPH 1: LOSS vs ITERATIONS ----------
plt.figure(figsize=(7, 4))
plt.plot(hist["loss"], color="blue")
plt.xlabel("Iterations"); plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations (learning rate = 0.01)")
plt.grid(True); plt.savefig("graph1_loss_vs_iterations.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- 5. GRAPH 2: ACTUAL vs PREDICTED ----------
plt.figure(figsize=(7, 4))
plt.scatter(X, Y, color="steelblue", label="Actual data points")
x_line = np.linspace(X.min(), X.max(), 100)
plt.plot(x_line, w * x_line + b, color="red", linewidth=2,
         label=f"Fitted line: y = {w:.2f}x + {b:.2f}")
plt.xlabel("X (input)"); plt.ylabel("Y (target)")
plt.title("Actual Data vs Gradient Descent Regression Line")
plt.legend(); plt.grid(True)
plt.savefig("graph2_actual_vs_predicted.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- 6. EXPERIMENT: DIFFERENT LEARNING RATES ----------
rates = [0.001, 0.01, 0.1]
results = {}
for lr in rates:
    results[lr] = gradient_descent(X, Y, lr, iterations)

print("\nLearning rate comparison (1000 iterations)")
print(f"{'LR':>7} {'Iters':>6} {'Final Loss':>14} {'Weight':>10} {'Bias':>10}")
for lr in rates:
    w_lr, b_lr, h = results[lr]
    print(f"{lr:>7} {len(h['loss'])-1:>6} {h['loss'][-1]:>14.4f} {w_lr:>10.4f} {b_lr:>10.4f}")

# Loss at selected iterations, to describe convergence
print("\nLoss at selected iterations")
for lr in rates:
    h = results[lr][2]
    print(lr, [round(float(h['loss'][i]), 3) if i < len(h['loss']) else None for i in (0, 10, 100, 500, 1000)])

plt.figure(figsize=(7, 4))
for lr in rates:
    plt.plot(results[lr][2]["loss"], label=f"learning rate = {lr}")
plt.yscale("log")
plt.xlabel("Iterations"); plt.ylabel("Loss (MSE, log scale)")
plt.title("Loss Curves for Different Learning Rates")
plt.legend(); plt.grid(True)
plt.savefig("graph3_learning_rate_comparison.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- 7. CHECK: exact answer from NumPy (verification only) ----------
print("\nVerification with np.polyfit (not used for training):", np.polyfit(X, Y, 1))
# ---------- 8. BONUS: a too-large rate (0.25) to show divergence ----------
w_big, b_big, h_big = gradient_descent(X, Y, 0.25, 50)
print("lr=0.25 loss at iterations 0,1,2,5,10:", [float(f"{h_big['loss'][i]:.4g}") for i in (0,1,2,5,10) if i < len(h_big['loss'])])
