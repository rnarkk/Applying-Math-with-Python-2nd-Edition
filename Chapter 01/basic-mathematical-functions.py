import math
import jax.numpy as np

np.sqrt(4)  # 2.0

theta = math.pi / 4
np.cos(theta)  # 0.7071067811865476
np.sin(theta)  # 0.7071067811865475
np.tan(theta)  # 0.9999999999999999

np.arcsin(-1)  # -1.5707963267948966
np.arccos(-1)  # 3.141592653589793
np.arctan(1)  # 0.7853981633974483

np.log(10)  # 2.302585092994046
np.log10(10)  # 1.0

math.gamma(5)  # 24.0
math.erf(2)  # 0.9953222650189527

math.comb(5, 2)  # 10
math.factorial(5)  # 120

np.gcd(2, 4) # 2
np.gcd(2, 3) # 1

nums = [0.1] * 10  # list containing 0.1 ten times
np.sum(nums)  # 0.9999999999999999
math.fsum(nums)  # 1.0
