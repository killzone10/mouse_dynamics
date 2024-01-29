import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

## Margit Antal1 Elod Egyed-Zsigmond Intrusion Detection Using Mouse Dynamics ˝

# HELPER FUNCTIONS ## 
def mean(array, start, stop):
    avg = 0
    amount = 0
    for i in range(start, stop):
        avg += array[i]
        amount += 1
    if amount == 0:
        return 0
    return avg / amount

# [start, stop)
def stdev(array, start, stop):
    m = mean(array, start, stop)
    n = stop - start
    if n-1 <= 0:
        return 0
    s = 0

    for i in range(start+1, stop):
        s += (array[ i ]-m)*(array[i]-m)
    return math.sqrt(s/(n-1))

# [start, stop)
def maximum(array, start, stop):
    n = stop - start
    if n-1 <= 0:
        return 0
    maxValue = array[start]
    for i in range(start, stop):
        if array[i] > maxValue:
            maxValue = array[ i ]
    return maxValue

# [start, stop)
def min_not_null(array, start, stop):
    n = stop - start
    if n-1 <= 0:
        return 0
    minValue = array[stop-1]
    for i in range(start, stop):
        if array[i] != 0 and array[i] < minValue:
            minValue = array[ i ]
    return minValue

def containsNull ( x ):
    for i in range(0,len(x)):
        if x[i] == 0:
            return True
    return False



def largestDeviation(x, y):
    n = len( x )
    #     line (x_0,y_0) and (x_n-1,y_n-1): ax + by + c
    a = float(x[n-1]) - float(x[0])
    b = float(y[0]) - float(y[n-1])
    c = float(x[0]) * float(y[n-1]) - float(x[n-1]) * float(y[0])
    max = 0
    den = math.sqrt(a*a+b*b)
    for i in range(1,n-1):
    #     distance x_i,y_i from line
        d = math.fabs(a*float(x[i])+b*float(y[i])+c)
        if d > max:
            max = d
    if den > 0:
        max /= den
    return max


# directions: 0..7
# Ahmed & Traore, IEEE TDSC2007
def     computeDirection(theta):
    direction = 0
    if 0 <= theta < math.pi / 4:
        direction = 0
    if math.pi / 4 <= theta < math.pi / 2:
        direction = 1
    if math.pi / 2 <= theta < 3 * math.pi / 4:
        direction = 2
    if 3 * math.pi / 4 <= theta < math.pi:
        direction = 3
    if -math.pi / 4 <= theta < 0:
        direction = 7
    if -math.pi / 2 <= theta < -math.pi / 4:
        direction = 6
    if -3 * math.pi / 4 <= theta < -math.pi / 2:
        direction = 5
    if -math.pi <= theta < -3 * math.pi / 4:
        direction = 4
    return direction



# directions: 0..7
# Chao Shen, TIFS, 2013-as cikk alapjan
# def computeDirection(theta):
#     direction = 0
#
#     if -math.pi / 8 <= theta < 0:
#         direction = 0
#     if 0 <= theta < math.pi / 8:
#         direction = 0
#
#
#     if math.pi/8 <= theta < 3*math.pi / 8:
#         direction = 1
#
#     if 3*math.pi / 8 <= theta < 5*math.pi / 8:
#         direction = 2
#
#     if 5*math.pi / 8 <= theta < 7 * math.pi / 8:
#         direction = 3
#
#     if 7*math.pi / 8 <= theta < math.pi:
#         direction = 4
#     if -math.pi <= theta <-7* math.pi/8:
#         direction = 4
#
#     if -7 * math.pi / 8 <= theta < -5*math.pi/8:
#         direction = 5
#
#     if -5 *math.pi / 8 <= theta < -3 *math.pi / 8:
#         direction = 6;
#
#     if -3 * math.pi / 8 <= theta < -math.pi / 8:
#         direction = 7;
#
#
#     return direction
