import numpy as np
from numpy import linalg as la
from helper import *
# PART A
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
X1, z1 = make_XZ(lambda1, M1, v1)
# print(np.shape(X1))
X2, z2 = make_XZ(lambda2, M2, v2)
# PART B
# print(X1)
mean1_est = maximum_likelihood_M(X1)
mean2_est = maximum_likelihood_M(X2)
sigma1_est = maximum_likelihood_Sigma(X1, mean1_est)
sigma2_est = maximum_likelihood_Sigma(X2, mean2_est)
plot_ML_mean(X1[0, :], M1[0, :], X2[0, :], M2[0, :], 'x1')
plot_ML_sigma(X1[0, :], M1[0, :], S1[0, 0], X2[0, :], M2[0, :], S2[0, 0], 'x1')
mean1_bayes_est = bayesian_learner(X1, S1)
mean2_bayes_est = bayesian_learner(X2, S2)
print(mean1_est)
print(mean1_bayes_est)
plot_bayes_mean(X1, M1, S1, X2, M2, S2, 'x1')
# ############################################################################
# PART C
pdf1, range1, mean11, var11 = parzen_window(X1[0, :])
print('class 1 mean in x1: ' + str(mean11) + ' and variance: ' + str(var11))
pdf2, range2, mean21, var21 = parzen_window(X2[0, :])
print('class 2 mean in x1: ' + str(mean21) + ' and variance: ' + str(var21))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf1, range1, pdf2, range2, 'x1')
# print(maximum_likelihood_Sigma(X1, mean1_est))
pdf12, range12, mean12, var12 = parzen_window(X1[1, :])
print('class 1 mean in x2: ' + str(mean12) + ' and variance: ' + str(var12))
pdf22, range22, mean22, var22 = parzen_window(X2[1, :])
print('class 2 mean in x2: ' + str(mean22) + ' and variance: ' + str(var22))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf12, range12, pdf22, range22, 'x2')
pdf13, range13, mean13, var13 = parzen_window(X1[2, :])
print('class 1 mean in x3: ' + str(mean13) + ' and variance: ' + str(var13))
pdf23, range23, mean23, var23 = parzen_window(X2[2, :])
print('class 3 mean in x3: ' + str(mean23) + ' and variance: ' + str(var23))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf13, range13, pdf23, range23, 'x3')
# #############################################################################
# PART D
# #############################################################################
# ML discrim
a = ((np.linalg.inv(sigma2_est) - np.linalg.inv(sigma1_est)) / 2)
b = mean1_est.transpose() @ np.linalg.inv(sigma1_est) - mean2_est.transpose() @ np.linalg.inv(sigma2_est)
c = np.log(np.linalg.det(sigma2_est) / np.linalg.det(sigma1_est))
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
plt.title("X1 -- X2 with discriminant ML")
plt.legend(loc=1)
plt.xlabel("X1")
plt.ylabel("X2")
minimum = min(min(min(X1[0, :]), min(X2[0, :])), min(min(X1[1, :]), min(X2[1, :])))
maximum = max(max(max(X1[0, :]), max(X2[0, :])), max(max(X1[1, :]), max(X2[1, :])))
plt.axis([minimum - 1, maximum + 1, minimum - 1, maximum + 1])
plt.show()
# #######################################################################
a = ((np.linalg.inv(S2) - np.linalg.inv(S1)) / 2)
b = mean1_bayes_est.transpose() @ np.linalg.inv(S1) - mean2_bayes_est.transpose() @ np.linalg.inv(S2)
c = np.log(np.linalg.det(S2) / np.linalg.det(S1))
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
# ###############################################################################
mean1_parzen = np.array([[mean11], [mean12], [mean13]])
mean2_parzen = np.array([[mean21], [mean22], [mean23]])
S1_parzen = np.array([[var11, 0, 0], [0, var12, 0], [0, 0, var13]])
S2_parzen = np.array([[var21, 0, 0], [0, var22, 0], [0, 0, var23]])
a = ((np.linalg.inv(S2_parzen) - np.linalg.inv(S1_parzen)) / 2)
b = mean1_bayes_est.transpose() @ np.linalg.inv(S1_parzen) - mean2_bayes_est.transpose() @ np.linalg.inv(S2_parzen)
c = np.log(np.linalg.det(S2_parzen) / np.linalg.det(S1_parzen))
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
# ############################################################################
# PART E
M_test_pts1 = np.array([[3], [1], [4]])
S_test_pts1 = S1
w_test1, v_test1 = np.linalg.eig(S_test_pts1)
lambda_test1 = np.diag(w_test1)

test_pts1, zt1 = make_XZ(lambda_test1, M_test_pts1, v_test1)

M_test_pts2 = np.array([[3], [1], [4]])
S_test_pts2 = S2
w_test2, v_test2 = np.linalg.eig(S_test_pts2)
lambda_test2 = np.diag(w_test2)

test_pts2, zt2 = make_XZ(lambda_test2, M_test_pts2, v_test2)

all_pts1 = np.concatenate((X1, test_pts1), axis=1)
# print("shape of all_pts1 is: " + str(np.shape(all_pts1)))
# print(all_pts1[:, 0:2:1])
all_pts2 = np.concatenate((X2, test_pts2), axis=1)
# print("shape of all_pts2 is: " + str(np.shape(all_pts2)))

# I'm pretty lazy, let's see if i can just procedurally do this

training1 = all_pts1[:, 0:40]
training2 = all_pts2[:, 0:40]
print("shape of training1 is: " + str(np.shape(training1)))
mean1_est = maximum_likelihood_M(training1)
mean2_est = maximum_likelihood_M(training2)
sigma1_est = maximum_likelihood_Sigma(training1, mean1_est)
sigma2_est = maximum_likelihood_Sigma(training2, mean2_est)
# print("here")
mean1_bayes_est = bayesian_learner(training1, S1)
mean2_bayes_est = bayesian_learner(training2, S2)
# print("here")
pdf1, range1, mean11, var11 = parzen_window(training1[0, :], pts=39)
# print("here")
pdf2, range2, mean21, var21 = parzen_window(training2[0, :], pts=39)
pdf1, range1, mean12, var12 = parzen_window(training1[1, :], pts=39)
pdf2, range2, mean22, var22 = parzen_window(training2[1, :], pts=39)
pdf1, range1, mean13, var13 = parzen_window(training1[2, :], pts=39)
pdf2, range2, mean23, var23 = parzen_window(training2[2, :], pts=39)
mean1_parzen = np.array([[mean11], [mean12], [mean13]])
mean2_parzen = np.array([[mean21], [mean22], [mean23]])
S1_parzen = np.array([[var11, 0, 0], [0, var12, 0], [0, 0, var13]])
S2_parzen = np.array([[var21, 0, 0], [0, var22, 0], [0, 0, var23]])

c1t, c1f, c2t, c2f, acc1, acc2 = classify(training1, training2, sigma1_est, sigma2_est, mean1_est, mean2_est)

print("ML-training Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
      ", accuracy of " + str(acc1))
print("ML-training Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
      ", accuracy of " + str(acc2))

c1t, c1f, c2t, c2f, acc1, acc2 = classify(training1, training2, S_test_pts1, S_test_pts2, mean1_bayes_est, mean2_bayes_est)

print("Bayes-training Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
      ", accuracy of " + str(acc1))
print("Bayes-training Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
      ", accuracy of " + str(acc2))

c1t, c1f, c2t, c2f, acc1, acc2 = classify(training1, training2, S1_parzen, S1_parzen, mean1_parzen, mean2_parzen)

print("Parzen-training Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
      ", accuracy of " + str(acc1))
print("Parzen-training Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
      ", accuracy of " + str(acc2))

# no longer training, now testing

for i in range(40, 400, 40):

    c1t, c1f, c2t, c2f, acc1, acc2 = classify(all_pts1[:,i:(i + 40)], all_pts2[:,i:(i + 40)], sigma1_est, sigma2_est, mean1_est, mean2_est)

    print("ML-testing block " + str(i/40) + " Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
          ", accuracy of " + str(acc1))
    print("ML-training block " + str(i/40) + " Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
          ", accuracy of " + str(acc2))

    c1t, c1f, c2t, c2f, acc1, acc2 = classify(all_pts1[:,i:(i + 40)], all_pts2[:,i:(i + 40)], S_test_pts1, S_test_pts2, mean1_bayes_est, mean2_bayes_est)

    print("Bayes-training block " + str(i/40) + " Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
          ", accuracy of " + str(acc1))
    print("Bayes-training block " + str(i/40) + " Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
          ", accuracy of " + str(acc2))

    c1t, c1f, c2t, c2f, acc1, acc2 = classify(all_pts1[:,i:(i + 40)], all_pts2[:,i:(i + 40)], S1_parzen, S1_parzen, mean1_parzen, mean2_parzen)

    print("Parzen-training block " + str(i/40) + " Is Class 1: " + str(c1t) + ", Not Class 1: " + str(c1f) +
          ", accuracy of " + str(acc1))
    print("Parzen-training block " + str(i/40) + " Is Class 1: " + str(c2f) + ", not Class 1: " + str(c2t) +
          ", accuracy of " + str(acc2))
# Part F
# ######################################################################################################################

# PART A
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
X1, z1 = make_XZ(lambda1, M1, v1)
# print(np.shape(X1))
X2, z2 = make_XZ(lambda2, M2, v2)
X1, M1, S1, X2, M2, S2 = two_class_diag(X1, M1, S1, X2, M2, S2)
# PART B
# print(X1)
mean1_est = maximum_likelihood_M(X1)
mean2_est = maximum_likelihood_M(X2)
sigma1_est = maximum_likelihood_Sigma(X1, mean1_est)
sigma2_est = maximum_likelihood_Sigma(X2, mean2_est)
plot_ML_mean(X1[0, :], M1[0, :], X2[0, :], M2[0, :], 'x1')
plot_ML_sigma(X1[0, :], M1[0, :], S1[0, 0], X2[0, :], M2[0, :], S2[0, 0], 'x1')
mean1_bayes_est = bayesian_learner(X1, S1)
mean2_bayes_est = bayesian_learner(X2, S2)
print(mean1_est)
print(mean1_bayes_est)
plot_bayes_mean(X1, M1, S1, X2, M2, S2, 'x1')
# ############################################################################
# PART C
pdf1, range1, mean11, var11 = parzen_window(X1[0, :])
print('class 1 mean in x1: ' + str(mean11) + ' and variance: ' + str(var11))
pdf2, range2, mean21, var21 = parzen_window(X2[0, :])
print('class 2 mean in x1: ' + str(mean21) + ' and variance: ' + str(var21))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf1, range1, pdf2, range2, 'x1')
# print(maximum_likelihood_Sigma(X1, mean1_est))
pdf12, range12, mean12, var12 = parzen_window(X1[1, :])
print('class 1 mean in x2: ' + str(mean12) + ' and variance: ' + str(var12))
pdf22, range22, mean22, var22 = parzen_window(X2[1, :])
print('class 2 mean in x2: ' + str(mean22) + ' and variance: ' + str(var22))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf12, range12, pdf22, range22, 'x2')
pdf13, range13, mean13, var13 = parzen_window(X1[2, :])
print('class 1 mean in x3: ' + str(mean13) + ' and variance: ' + str(var13))
pdf23, range23, mean23, var23 = parzen_window(X2[2, :])
print('class 3 mean in x3: ' + str(mean23) + ' and variance: ' + str(var23))
# plot_parzen_window_1(pdf1, range1, 'x1')
plot_parzen_window_2(pdf13, range13, pdf23, range23, 'x3')
# #############################################################################
# PART D
# #############################################################################
# ML discrim
a = ((np.linalg.inv(sigma2_est) - np.linalg.inv(sigma1_est)) / 2)
b = mean1_est.transpose() @ np.linalg.inv(sigma1_est) - mean2_est.transpose() @ np.linalg.inv(sigma2_est)
c = np.log(np.linalg.det(sigma2_est) / np.linalg.det(sigma1_est))
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
plt.title("X1 -- X2 with discriminant ML")
plt.legend(loc=1)
plt.xlabel("X1")
plt.ylabel("X2")
minimum = min(min(min(X1[0, :]), min(X2[0, :])), min(min(X1[1, :]), min(X2[1, :])))
maximum = max(max(max(X1[0, :]), max(X2[0, :])), max(max(X1[1, :]), max(X2[1, :])))
plt.axis([minimum - 1, maximum + 1, minimum - 1, maximum + 1])
plt.show()
# #######################################################################
a = ((np.linalg.inv(S2) - np.linalg.inv(S1)) / 2)
b = mean1_bayes_est.transpose() @ np.linalg.inv(S1) - mean2_bayes_est.transpose() @ np.linalg.inv(S2)
c = np.log(np.linalg.det(S2) / np.linalg.det(S1))
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
# ###############################################################################
mean1_parzen = np.array([[mean11], [mean12], [mean13]])
mean2_parzen = np.array([[mean21], [mean22], [mean23]])
S1_parzen = np.array([[var11, 0, 0], [0, var12, 0], [0, 0, var13]])
S2_parzen = np.array([[var21, 0, 0], [0, var22, 0], [0, 0, var23]])
a = ((np.linalg.inv(S2_parzen) - np.linalg.inv(S1_parzen)) / 2)
b = mean1_bayes_est.transpose() @ np.linalg.inv(S1_parzen) - mean2_bayes_est.transpose() @ np.linalg.inv(S2_parzen)
c = np.log(np.linalg.det(S2_parzen) / np.linalg.det(S1_parzen))
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