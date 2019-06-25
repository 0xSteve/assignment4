'''This is the main script for the assignment.'''

import numpy as np
from numpy import linalg as la
from helper import *

# I made some mistakes with my assignment 2, so I've updated my
# diagonalizing code from some guaranteed correct code from Omar.
# The parts that are from assignment 2 have been corrected thanks to Omar's
# help.

# np.array(multi_normal_vector(sizeof, 3))
# Okay, this time get M1 and M2 as column vectors...
M1 = np.array([[3], [1], [4]])
M2 = np.array([[-3], [1], [-4]])
# print(np.shape(m1))
alpha = 0.1
beta = 0.2
a = 2
b = 3
c = 4

# from previous assignment.
S1 = [[a * a, beta * a * b, alpha * a * c],
          [beta * a * b, b * b, beta * b * c],
          [alpha * a * c, beta * b * c, c * c]]
S1 = np.array(np.round(S1, 6))
S2 = [[c * c, alpha * b * c, beta * a * c],
          [alpha * b * c, b * b, alpha * a * b],
          [beta * a * c, alpha * a * b, a * a]]
S2 = np.array(np.round(S2, 6))
print("SIGMA 1 \n" + str(S1))
print("SIGMA 2 \n" + str(S2))

w1, v1 = np.linalg.eig(S1)
lambda1 = np.diag(w1)
w2, v2 = np.linalg.eig(S2)
lambda2 = np.diag(w2)

print("X1 Eigenvalues \n " + str(lambda1))
print("X2 Eigenvalues \n " + str(lambda2))
# I really don't know why it was having a size error when I did this
# individually. Figure this out later. Tomorrow perhaps?
X1, z1 = make_XZ(lambda1, M1, v1)
# print(np.shape(X1))
X2, z2 = make_XZ(lambda2, M2, v2)
# print("z1 is \n" + str(z1))
# print("X1 is \n" + str(X1))
# print("z2 is \n" + str(z2))
# print("X2 is \n" + str(X2))
# Now to plot... Perhaps make a graphing functing?
nice_plot(X1, X2, 1, 2, 'X1', 'X2')
nice_plot(X1, X2, 1, 3, 'X1', 'X3')

a = ((np.linalg.inv(S2) - np.linalg.inv(S1)) / 2)
b = M1.transpose() @ np.linalg.inv(S1) - M2.transpose() @ np.linalg.inv(S2)
# Don't specify base for math.log base e, (ln), np.log is base e
c = np.log(np.linalg.det(S2) / np.linalg.det(S1))
# print("a is \n " + str(a))
# print("b is \n " + str(b))
# print("c is \n " + str(c))
x_axis_pts = []
r1 = []
r2 = []

ax2 = a[1, 1]
for i in range(-12, 10, 1):
    x_axis_pts.append(i)
    bx = (a[0, 1] * i) + (a[1, 0] * i) + b[0, 1]
    const = (a[0, 0] * np.math.pow(i, 2)) + (b[0, 0] * i) + c

    roots = np.roots([ax2, bx, const])
    # I don't want to think about the shape anymore.
    r1.append(roots[0])
    r2.append(roots[1])
# print(r1)
# print(r2)
plt.plot(X1[0, :], X1[1, :], 'y.', label="Class One")
plt.plot(X2[0, :], X2[1, :], 'g.', label="Class Two")
plt.plot(x_axis_pts, r1, 'r-', label="Discriminant Root 1")
plt.plot(x_axis_pts, r2, 'b-', label="Discriminant Root 2")
plt.title("X1 -- X2 with discriminant")
plt.legend(loc=1)
plt.xlabel("X1")
plt.ylabel("X2")
# Okay I seriously need to add axis
minimum = min(min(min(X1[0, :]), min(X2[0, :])), min(min(X1[1, :]), min(X2[1, :])))
maximum = max(max(max(X1[0, :]), max(X2[0, :])), max(max(X1[1, :]), max(X2[1, :])))
plt.axis([minimum - 1, maximum + 1, minimum - 1, maximum + 1])
plt.show()

x_axis_pts = []
r1 = []
r2 = []

ax2 = a[2][2]
for i in range(-12, 12, 1):
    x_axis_pts.append(i)
    bx = a[0, 2] * i + a[2, 0] * i + b[0, 2]
    const = a[0, 0] * np.math.pow(i, 2) + b[0, 0] * i + c

    # use numpy to get roots from here
    roots = np.roots([ax2, bx, const])
    # I don't want to think about the shape anymore.
    r1.append(roots[0])
    r2.append(roots[1])
# print(r1)
plt.plot(X1[0, :], X1[2, :], 'y.', label="Class One")
plt.plot(X2[0, :], X2[2, :], 'g.', label="Class Two")
plt.plot(x_axis_pts, r1, 'r-', label="Discriminant Root 1")
plt.plot(x_axis_pts, r2, 'b-', label="Discriminant Root 2")
plt.title("X1 -- X3 with discriminant")
plt.legend(loc=1)
plt.xlabel("X1")
plt.ylabel("X3")
# Okay I seriously need to add axis
minimum = min(min(min(X1[0, :]), min(X2[0, :])), min(min(X1[2, :]), min(X2[2, :])))
maximum = max(max(max(X1[0, :]), max(X2[0, :])), max(max(X1[2, :]), max(X2[2, :])))
plt.axis([minimum - 1, maximum + 1, minimum - 1, maximum + 1])
plt.show()

# One root doesn't show up due to scale.

# okay, for this part i can use my function!

trd1, _ = make_XZ(lambda1, M1, v1)
trd2, _ = make_XZ(lambda2, M2, v2)

V1, Mv1, Sv1, V2, Mv2, Sv2 = two_class_diag(X1, M1, S1, X2, M2, S2)
trd_d1, mtrd1, strd1, trd_d2, mtrd2, strd2 = two_class_diag(V1, Mv1, Sv1, V2, Mv2, Sv2)

c1t, c1f, c2t, c2f, acc1, acc2 = classify(trd1, trd2, S1, S2, M1, M2)

print("Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
      ", accuracy of " + str(acc1))
print("Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
      ", accuracy of " + str(acc2))


Sv1 = np.round(Sv1, 6)
Sv2 = np.round(Sv2, 6)
nice_plot(V1, V2, 1, 2, 'V1', 'V2')
nice_plot(V1, V2, 1, 3, 'V1', 'V3')
print(Sv2)
print(Sv1)

a = (np.linalg.inv(Sv2) - np.linalg.inv(Sv1)) / 2
b = Mv1.transpose() @ np.linalg.inv(Sv1) - Mv2.transpose() @ np.linalg.inv(Sv2)
# Don't specify base for math.log base e, (ln), np.log is base e
c = np.log(np.linalg.det(Sv2) / np.linalg.det(Sv1))
# print("a is \n " + str(a))
# print("b is \n " + str(b))
# print("c is \n " + str(c))
x_axis_pts = []
r1 = []
r2 = []

ax2 = a[1, 1]
for i in range(-10, 10, 1):
    x_axis_pts.append(i)
    bx = (a[0, 1] * i) + (a[1, 0] * i) + b[0, 1]
    const = (a[0, 0] * np.math.pow(i, 2)) + (b[0, 0] * i) + c

    roots = np.roots([ax2, bx, const])
    # I don't want to think about the shape anymore.
    r1.append(roots[0])
    r2.append(roots[1])
# print(r1)
# print(r2)
plt.plot(V1[0, :], V1[1, :], 'y.', label="Class One")
plt.plot(V2[0, :], V2[1, :], 'g.', label="Class Two")
plt.plot(x_axis_pts, r1, 'r-', label="Discriminant Root 1")
plt.plot(x_axis_pts, r2, 'b-', label="Discriminant Root 2")
plt.title("V1 -- V2 with discriminant")
plt.legend(loc=1)
plt.xlabel("V1")
plt.ylabel("V2")
# Okay I seriously need to add axis
minimum = min(min(min(V1[0, :]), min(V2[0, :])), min(min(V1[1, :]), min(V2[1, :])))
maximum = max(max(max(V1[0, :]), max(V2[0, :])), max(max(V1[1, :]), max(V2[1, :])))
plt.axis([minimum - 5, maximum + 5, minimum - 5, maximum + 5])
plt.show()

x_axis_pts = []
r1 = []
r2 = []

ax2 = a[2][2]
for i in range(-10, 10, 1):
    x_axis_pts.append(i)
    bx = a[0, 2] * i + a[2, 0] * i + b[0, 2]
    const = a[0, 0] * np.math.pow(i, 2) + b[0, 0] * i + c

    # use numpy to get roots from here
    roots = np.roots([ax2, bx, const])
    # I don't want to think about the shape anymore.
    r1.append(roots[0])
    r2.append(roots[1])
# print(r1)
plt.plot(V1[0, :], V1[2, :], 'y.', label="Class One")
plt.plot(V2[0, :], V2[2, :], 'g.', label="Class Two")
plt.plot(x_axis_pts, r1, 'r-', label="Discriminant Root 1")
plt.plot(x_axis_pts, r2, 'b-', label="Discriminant Root 2")
plt.title("V1 -- V3 with discriminant")
plt.legend(loc=1)
plt.xlabel("V1")
plt.ylabel("V3")
# Okay I seriously need to add axis
minimum = min(min(min(V1[0, :]), min(V2[0, :])), min(min(V1[2, :]), min(V2[2, :])))
maximum = max(max(max(V1[0, :]), max(V2[0, :])), max(max(V1[2, :]), max(V2[2, :])))
plt.axis([minimum - 5, maximum + 5, minimum - 5, maximum + 5])
plt.show()

# There might be a real reason why this isn't showing up correctly...

c1t, c1f, c2t, c2f, acc1, acc2 = classify(trd_d1, trd_d2, strd1, strd2, mtrd1, mtrd2)

print("Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
      ", accuracy of " + str(acc1))
print("Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
      ", accuracy of " + str(acc2))
