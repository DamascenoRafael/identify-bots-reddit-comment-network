import numpy as np

def pdf(array):
    values, counts = np.unique(array, return_counts=True)
    pdf = counts / len(array)
    return values, pdf

def cdf(array):
    values, v_pdf = pdf(array)
    cdf = np.cumsum(v_pdf)
    cdf[-1] = 1
    return values, np.array(cdf)

def ccdf(array):
    values, v_cdf = cdf(array)
    ccdf = 1 - v_cdf
    ccdf = np.concatenate(([1], ccdf[:-1]))
    return values, ccdf
