'''
multi-layer perceptron
'''

from perceptron import bias_AND,bias_NAND,bias_OR
def XOR(x1,x2):
    s1 = bias_NAND(x1,x2)
    s2 = bias_OR(x1,x2)
    y = bias_AND(s1,s2)
    return y

def half_adder(x1,x2):
    _sum = XOR(x1,x2)
    carry = bias_AND(x1,x2)
    return _sum,carry

def full_adder(x1,x2,carry):
    sum1 , carry1 = half_adder(x1,x2)
    sum2 , carry2 = half_adder(sum1,carry)
    return sum2, bias_OR(carry1,carry2)