from mpmath import mp


def gamma(x):
    '''
    Returns the gamma function value of any complex number as defined by the
    improper integral derived by Daniel Bernoulli.
    '''
    if x.real <= 0:
        if x.imag == 0 and float(x.real) == int(x.real):
            return mp.inf
        else:
            return mp.pi / (mp.sin(mp.pi * x) * gamma(1 - x))
    else:
        result, _ = mp.quad(lambda t: t ** (x - 1) * mp.exp(-t), [0, mp.inf],
                            error=True)
        return result


def factorial(x):
    '''
    Returns the factorial of any positive integer by taking the gamma function
    of the input plus 1, while stating if the input is not appropriate.
    '''
    if x < 0 or type(x) is not int:
        return "The input must be a positive integer."
    else:
        return int(gamma(x + 1))


def dt(t, v):
    '''
    Returns the probability P(T=t) under a t-distribution given an input random
    variable t, and the positive number of degrees of freedom v.
    '''
    if v < 0:
        return "The number of degrees of freedom must be positive."
    return ((gamma((v + 1) / 2) / (mp.sqrt(mp.pi * v) * gamma(v / 2))) *
            (1 + ((t ** 2) / v)) ** (-(v + 1) / 2))


def pt(q, v, lower_tail=True, two_sided=False):
    '''
    Returns the cumulative probabilities P(T<=t) or P(T>t) (when lower_tail is
    False) under a t-distribution given an input random variable q, and the
    positive number of degrees of freedom v. Works for both one-sided and
    two-sided tests.
    '''
    if v < 0:
        return "The number of degrees of freedom must be positive."
    if lower_tail:
        tail = [-mp.inf, q]
    else:
        tail = [q, mp.inf]
    p, _ = mp.quad(lambda t: dt(t, v), tail, error=True)
    if two_sided:
        return 2 * p
    return p


def dchisq(x, k):
    '''
    Returns the probability P(X=x) under a chi-squared-distribution given an
    input random variable x, and the positive number of degrees of freedom k.
    '''
    if k < 0:
        return "The number of degrees of freedom must be positive."
    elif x <= 0:
        return 0
    return ((x ** ((k / 2) - 1)) * (mp.e ** (-x / 2))) / ((2 ** (k / 2)) *
                                                          gamma(k / 2))


def pchisq(q, k, lower_tail=True):
    '''
    Returns the cumulative probabilities P(X<=x) or P(X>x) (when lower_tail is
    False) under a chi-squred-distribution given an input random variable q,
    and the positive number of degrees of freedom k.
    '''
    if k < 0:
        return "The number of degrees of freedom must be positive."
    if lower_tail:
        tail = [-mp.inf, q]
    else:
        tail = [q, mp.inf]
    p, _ = mp.quad(lambda t: dchisq(t, k), tail, error=True)
    return p


def main():
    print(gamma(-3.4 + 2j))
    print(factorial(4))
    print(dt(1.96, 5))
    print(pt(8 / 9, 8))
    print(dchisq(5, 10))
    print(pchisq(2.98, 10))


if __name__ == '__main__':
    main()
