# Module Declarations
import numpy as np
import matplotlib.pyplot as plt
import itertools
###########


# Useful utilities
def load_data(input_path, 
            output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y
def polynomial(i, 
                x):
    return x ** (i - 1)
def phi_gen(D,
            x):
    phi = [[0] * D for i in range(len(x))]
    for i, j in enumerate(x):
        phi[i] = [polynomial(d, j) for d in range(1, D + 1)]
    return phi
def w_mle_gen(phi, 
                y):
    inner = np.dot(np.transpose(phi), phi)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w
def compute_basis(w, 
                x,
                d):
        total = 0
        for i in range(d):
            total += w[i] * polynomial(i+1, x)
        return total
def mean_square_error(D, 
                    w,
                    y,
                    x):
    mse = 0
    for j in range(len(y)):
        mse += (y[j] - compute_basis(w, x[j], D))**2
    return mse / len(y)
def generate_w_map(phi,
                    y, 
                    d):
    lam = 0.001
    identity = lam * np.identity(d)
    
    phi_t = np.dot(np.transpose(phi), phi)
    inner = np.add(identity, phi_t)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

plt.plot(x, y, "x", color="green")
plt.show()

# Step 2
num_basis = 20 # assumption of problem
mses = np.zeros(num_basis)

for D in range(1, num_basis + 1):
    phi = phi_gen(D, x)
    w = w_mle_gen(phi, y)
    mses[D - 1] = mean_square_error(D, w, y, x)
# PLOT
plt.plot(range(0, 21), np.concatenate(([None], mses)))
plt.title("MSE as a Function of D with MLE Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()


# Step 3
mses_mle = mses
num_basis = 20
mses = np.zeros(num_basis)
for D in range(1, num_basis + 1):
    phi = phi_gen(D, x)
    w = generate_w_map(phi, y, D)
    mses[D - 1] = mean_square_error(D, w, y, x)
# PLOT
plt.plot(range(0, 21), np.concatenate(([None], mses)), label="MAP", color="green")
plt.plot(range(0, 21), np.concatenate(([None], mses_mle)), label="MLE", color="red")
plt.legend()
plt.title("MSE as a Function of D with MAP and MLE Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()

# Step 4

for D in range(1, num_basis + 1):
    phi = phi_gen(D, x)
    w_mle = w_mle_gen(phi, y)
    w_map = generate_w_map(phi, y, D)

    x_new = np.linspace(min(x), max(x), 200)     # Create new x range for sampling more points

    pred_y_mle = np.zeros(len(x_new))
    pred_y_map = np.zeros(len(x_new))
    for i in range(len(x_new)):
        pred_y_mle[i] = compute_basis(w_mle, x_new[i], D)
        pred_y_map[i] = compute_basis(w_map, x_new[i], D)

    data_mle = sorted(itertools.izip(*[x_new, pred_y_mle]))
    mle_x, mle_y = list(itertools.izip(*data_mle))

    # PLOT
    plt.plot(mle_x, mle_y, "-", label="MLE", color="red")
    data_map = sorted(itertools.izip(*[x_new, pred_y_map]))
    map_x, map_y = list(itertools.izip(*data_map))
    plt.plot(map_x, map_y, "-", label="MAP", color="blue")
    plt.plot(x, y, "x", label="Data", color="green")
    plt.title("D = {}".format(D))
    plt.legend()
    plt.show()


# Step 5
fold_def = 10
avg = np.zeros(num_basis)

for i in range(1, num_basis + 1):
    mses = np.zeros(10)
    xy = np.random.permutation(zip(x, y))
    new_x = xy[:, 0]
    new_y = xy[:, 1]
    for j in range(0, len(new_x), fold_def):

        training = np.concatenate([new_x[0:j], new_x[j+fold_def : len(new_x)]])
        training_y = np.concatenate([new_y[0:j], new_y[j+fold_def: len(new_y)]])

        validation = new_x[j:j + fold_def]
        valid_y = new_y[j:j + fold_def]

        phi = phi_gen(i, training)
        w_map = generate_w_map(phi, training_y, i)

        mses[j/fold_def] = mean_square_error(i, w_map, valid_y, validation)

    avg[i-1] = np.mean(mses)

# PLOT
plt.plot(range(0, 21), np.concatenate(([None], avg)))
plt.title("Ten Fold Cross Validation Average MSE results")
plt.xlabel("D")
plt.ylabel("MSE")
plt.show()

# Step 6
x, y = load_data("dataset2_inputs.txt", "dataset2_outputs.txt")
# PLOT
plt.plot(x, y, "x", color="red")
plt.show()

# Step 7
