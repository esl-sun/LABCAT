import numpy as np

import labcat
# from temp_bind import labcatConfig

def init_pts_fn(d):
    return 2 * d+1

def forget_fn(d):
    return d * 7

# # Himmelblau function
# def f(x):
#     return (x[0] * x[0] + x[1] - 11) ** 2 + (x[0] + x[1] * x[1] - 7) ** 2

# # Zakharov function https://www.sfu.ca/~ssurjano/zakharov.html
# def f(x):
#     return (x[0] * x[0] + x[1] * x[1]) + (0.5 * x[0] + 0.5 * 2 * x[1]) ** 2 + (0.5 * x[0] + 0.5 * 2 * x[1]) ** 4

# # Beale function https://www.sfu.ca/~ssurjano/beale.html
# def f(x):
#     return (
#         (1.5 - x[0] + x[0] * x[1]) ** 2
#         + (2.25 - x[0] + x[0] * x[1] * x[1]) ** 2
#         + (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) ** 2
#     )

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


bound_config = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)}, 
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)}, 
    "temp" : {"type": "cat", "categories": ["c1", "c2"]},  
    "temp2" : {"type": "bool"},
    }

test = [{
    "max_depth": 5,
    "min_samples_split": 0.5,
    "temp": "c1",
    "temp2": False,
}, {
    "max_depth": 5,
    "min_samples_split": 0.5,
    "temp": "c1",
    "temp2": False,
}]

b = labcat.BoundsConfig()
b.parse_config(bound_config)
# b.add_discrete("d0", 10, -10)
# b.add_continuous(3, 0)
# b.add_categorical("c1", ["1", "2", "3"])
# b.add_continuous("d1", 5, -5)
# b.add_continuous_with_transform("d2", 1000, 0.01, "log")
# b.add_continuous(10, -10)
# b.add_continuous(10, -10)
b = b.build()
print(repr(b))
# print(b.parse(test))

b = labcat.Bounds(2, 5, -5)
print(repr(b))

alg = labcat.LABCATConfig(b)
alg.beta(0.5)
alg.init_fn(init_pts_fn)
alg.forget_fn(forget_fn)
alg.max_samples(500)
# alg.target_tol(1e-6)
alg = alg.build()

for i in range(50):
    x = alg.suggest()
    y = f(x)
    alg.observe(x, y)

alg = alg.set_target_fn(f)
alg.print_interval(25)
res = alg.run()

print(str(res))