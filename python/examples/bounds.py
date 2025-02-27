import numpy as np

import labcat

# Bounds can be constructed using the builder pattern, similar to the Rust example of bounds.rs
b = labcat.BoundsConfig()
b.add_discrete("d1", 10, -10)
b.add_categorical("d2", ["1", "2", "3"])
b.add_continuous("d3", 5, -5)
b.add_continuous_with_transform("d4", 1000, 0.01, "log")
b = b.build()

# Bounds can also be parsed from a dict directly
bound_config = {
    "d1": {"type": "int", "space": "linear", "range": (1, 15)}, 
    "d2": {"type": "real", "space": "logit", "range": (0.01, 0.99)}, 
    "d3" : {"type": "cat", "categories": ["c1", "c2"]},  
    "d4" : {"type": "bool"},
    }

b = labcat.BoundsConfig()
b.parse_config(bound_config)
b = b.build()

# Continuous bounds can easily be constructed using the following constructor, in this case, two-dimensional with upper bounds of 5.0 and lower bounds of -5.0 for each dimensions
b = labcat.Bounds(2, 5.0, -5.0)

# Bounds can be printed the console using
print(b)