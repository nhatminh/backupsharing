using PyPlot
using Distributions



a=zeros(2,3)
a[1,1]=1
a[1,2]=2
a[2,1]=3
a[2,2]=4
contour(a)
