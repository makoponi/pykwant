import math


def norm_pdf(x: float) -> float:
    """
    Docstring for norm_pdf
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def norm_cdf(x: float) -> float:
    """
    Docstring for norm_cdf
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
