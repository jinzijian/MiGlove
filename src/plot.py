import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
def plot(gr, gg, gb):
    dis = gg - gr
    res = []
    for i in range(len(gb)):
        tmp = gb[i] - gr
        res.append(tmp/dis)
    return res


gr = -0.0004489032758606805
gg = 0.0752736632194784
gb = [0.013084687292575836, 0.013927506282925606, -0.00441028405394819, -0.0015653735026717186, 0.0013018175959587097, 0.006844884819454617, 0.012481372389528487, -0.0010137726138863298, 0.008829125099711947, -0.00022927713063028123, 0.008355612452659342, 0.013729531317949295, 0.01703992486000061]
res = plot(gr, gg, gb)
length = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(res, length)
plt.show()
pylab.show()