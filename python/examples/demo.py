import numpy as np

import labcat

# Rosenbrock function
def f(x):
    A = 1
    B = 100
    total = 0
    for i in range(x.shape[0]-1):

        total += B * (x[i+1] - x[i]**2)**2 + (x[i]-A)**2
    
    return total

def objective(x):
    return np.apply_along_axis(f, 0, x)

# More options for constructing Bounds can be found in the bounds.py example
bounds = labcat.Bounds(2, 5.0, -5.0)

# Similar to the Rust example, the Python bindings follow the builder pattern
alg_config = labcat.LABCATConfig(bounds)

# Default values for the LABCAT algorithm
alg_config.beta(1.0 / bounds.dim())
alg_config.prior_sigma(0.1)

def init_pts_fn(d):
    return 2 * d + 1

alg_config.init_fn(init_pts_fn)

def forget_fn(d):
    return d * 7

alg_config.forget_fn(forget_fn)

# After configuring the algorithm, the build() command enables the use of the ask-tell interface
ask_tell = alg_config.build()

for i in range(20):
    x = ask_tell.suggest()
    y = f(x)
    ask_tell.observe(x, y)

# By providing the objective function callable using set_target_fn, the algorithm can now automatically query the objective funtion
auto = ask_tell.set_target_fn(f)

# Termination conditions follow the same logic as in the Rust example, ending iteration when the first condition is met
auto.max_samples(150)
auto.target_val(1e-6)
auto.target_tol(1e-9)
# Max time is given in seconds
auto.max_time(10)

auto.print_interval(25)

# The algorithm can be executed using the run() command
res = auto.run()

print(str(res))