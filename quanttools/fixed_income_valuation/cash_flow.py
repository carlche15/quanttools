import numpy as np



def amortization_amt(maturity, coupon, freq = "M"):
    # TODO: add frequency

    gamma = 1 + np.array(coupon)/ 1200
    return (gamma ** maturity * (gamma - 1) / (gamma ** maturity - 1))

print(3e5*amortization_amt([360,240],[3.5,4.5]))

