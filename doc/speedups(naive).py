import pylab

#this is for the naive implementations
s = [1,2,4,6,8,12,16,24,28,30,31,32,33,34,36,40 ,  48,64,96,128,192,256,384,512]
t = [2.91,4.59, 7.51,10.18,13.28,18.49,23.13,32.65,36.55,39.12,39.33,(40.92+39.98)/2,     24.81,23.53,23.32,24.48,   28.03,34.70,36.95,35.96,37.96,36.19,38.30,37.38]
pylab.title("Threads Per Block vs. Speedup (constant blocks 65535)")
pylab.ylabel("GPU Parallel Speedup")
pylab.xlabel("Number of Blocks")
pylab.scatter(s,t,label="datapoints om")
pylab.plot(s,t,label="offchip mem")

s = [1,2,3,4,5,6,8,10,12,16,20,24,28,30,32,34,36,37]
t = [8.250000,14.666667,22.111111,   29.333334,27.666666, 28.500000 ,28.000000,24.518518, 18.904762, 21.591837, 13.612245,16.070707,18.660000,  19.627451,20.403847,21.728155,23.155340,23.883495]
pylab.scatter(s,t,label="datapoints shared", color="green")
pylab.plot(s,t,label="shared mem")

pylab.grid(ydata = [i for i in range(0,544,32)],xdata = [i for i in range(0,544,32)])
pylab.legend(loc="best")
pylab.show()
