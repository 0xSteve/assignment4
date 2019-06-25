'''Some helper files for this work.'''

from random import uniform as rand
import numpy as np
from numpy import linalg as la
from math import *
import matplotlib.pyplot as plt


def normal_point(dimensions=3):
    '''Generate a Normally distributed point from a multivariate Normal
       distribution of a specified dimension, by default 3.'''
    x = []
    for i in range(dimensions):
        z = 0
        for j in range(12):
            a = rand(0, 1)
            z += a
        z -= 6
        x.append([z])
    x = np.array(x)
    return x

# THIS IDEA DID NOT WORK AT ALL! Somehow doing them together did?

# def make_Z(size=200, dimensions=3):
#     '''Create an array of Normally distributed points.'''
#     Z = normal_point(dimensions)  # Will become a non-empty numpy array.

#     for i in range(size - 1):
#         point = normal_point(dimensions)
#         # add the new point on axis 1 to create a new column.
#         Z = np.append(Z, point, axis=1)

#     return Z


# def make_X(Z, lambda_x, mean, eigvec, size=200, dimensions=3):
#     '''Given a standard normal vector Z, a mean, and eigen values and vectors of
#        variance, generate a translated normal vector X.'''

#     x = eigvec @ np.power(lambda_x, 0.5) @ Z[0, :] + mean

#     for i in range(1, size):
#         point = eigvec @ np.power(lambda_x, 0.5) @ Z[i] + mean
#         x = np.append(x, point, axis=1)

#     return x

def make_XZ(lambda_x, mean, eigvec, size=200, dimensions=3):
    '''Given a standard normal vector Z, a mean, and eigen values and vectors of
       variance, generate a translated normal vector X.'''
    z = normal_point(dimensions)

    x = eigvec @ np.power(lambda_x, 0.5) @ z + mean

    for i in range(1, size):
        pt = normal_point(dimensions)
        z = np.append(z, pt, axis=1)
        pt =  eigvec @ np.power(lambda_x, 0.5) @ pt + mean
        x = np.append(x, pt, axis=1)
    # ROFL Don't indent the return!!!!
    return x, z



def inv_sqrt(A):
    '''Perform the element wise inverse square root. Assumes m x n matrix.'''
    for i in range(len(A)):
        for j in range(len(A[1])):
            A[i][j] = math.pow(A[i][j], -0.5)

    return A


def two_class_discriminant(trd, sigma1, sigma2, mean1, mean2, p1=0.5, p2=0.5):
    '''Compute the two class discriminant for a given set of training data.'''
    typer = np.zeros(1)
    # If the training data and others are not np.array type make them np.array
    # type.
    if(type(trd) != type(typer)):
        trd = np.array(trd)

    if(type(sigma1) != type(typer)):
        trd = np.array(sigma1)

    if(type(sigma2) != type(typer)):
        trd = np.array(sigma2)

    if(type(mean1) != type(typer)):
        trd = np.array(mean1)

    if(type(mean2) != type(typer)):
        trd = np.array(mean2)

    # From our course notes
    # ax^2 +bx + c
    # xT a x => equivalent to x^2
    a = ((np.linalg.inv(sigma2) - np.linalg.inv(sigma1)) / 2)
    b = mean1.transpose() @ np.linalg.inv(sigma1) - mean2.transpose() @ np.linalg.inv(sigma2)
    # Don't specify base for math.log base e, (ln), np.log is base e
    c = np.math.log(p1 / p2) + np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))

    return (trd.transpose() @ a @ trd + b @ trd + c)


def classify(trd1, trd2, S1, S2, M1, M2, p1=0.5, p2=0.5):
    '''Do a 2 class classification problem.'''
    # I should make this an array but it is 4 am.
    c1_T = 0
    c2_T = 0
    c1_F = 0
    c2_F = 0

    for i in range(0, len(trd1[1, :])):
        discrim = two_class_discriminant(trd1[:, i], S1, S2, M1, M2, p1, p2)
        if(discrim > 0):
            c1_T += 1
        else:
            c1_F += 1

    for i in range(0, len(trd2[1, :])):
        discrim = two_class_discriminant(trd2[:, i], S1, S2, M1, M2, p1, p2)

        if(discrim < 0):
            c2_T += 1
        else:
            c2_F += 1

    acc1 = c1_T / (len(trd1[1, :]) - 1)
    acc2 = c2_T / (len(trd2[1, :]) - 1)
    return c1_T, c1_F, c2_T, c2_F, acc1, acc2

# Now to correct some problems with my diagonalization from last assignment. I
# figured it would be better to include it in my helper functions instead of
# making a lengthy setup in my main assignment. I think I may end up doing the
# same thing for plotting.
def two_class_diag(X1, M1, S1, X2, M2, S2):
    '''Solve the simultaneous diagonalization problem.'''
    # This part has the corrections to my assigment2 mistake. Mostly it was
    # an issue with dimensions in numpy. I'm used to dimensioning my vectors
    # MATLAB style, and I didn't want to be looping over and over.
    w1, v1 = np.linalg.eig(S1)
    w2, v2 = np.linalg.eig(S2)

    # make the intermediary Y...
    Y1 = v1.transpose() @ X1
    Y2 = v2.transpose() @ X2
    # Mean of Y
    My1 = v1.transpose() @ M1
    My2 = v2.transpose() @ M2
    # Mean of Z
    Mz1 = v1.transpose() @ My1
    Mz2 = v2.transpose() @ My2
    # Mean of V
    Mv1 = v1.transpose() @ Mz1
    Mv2 = v2.transpose() @ Mz2

    # Make Z
    # instead of using P1 and so forth just use the w1 and w2
    Z1 = np.diag(np.power(w1, -0.5)) @ v1.transpose() @ X1
    Z2 = np.diag(np.power(w2, -0.5)) @ v1.transpose() @ X2
    # Make Sz1, Sz2
    Sz1 = np.diag(np.power(w1, -0.5)) @ np.diag(w1) @ np.diag(np.power(w1, -0.5))
    Sz2 = np.diag(np.power(w1, -0.5)) @ v1.transpose() @ S2 @ v1 @ np.diag(np.power(w1, -0.5))
    # Now get the P overall
    Poa = 0 # stands for P OverAll

    # make V1 V2
    wz1, vz1 = np.linalg.eig(Sz1)
    wz2, vz2 = np.linalg.eig(Sz2)
    Sv1 = vz2.transpose() @ Sz1 @ vz2
    Sv2 = vz2.transpose() @ Sz2 @ vz2
    Poa = vz2.transpose() @ np.diag(np.power(w1, -0.5)) @ v1.transpose()
    V1 = Poa @ X1
    V2 = Poa @ X2
    # maybe return everything, or just V
    return V1, Mv1, Sv1, V2, Mv2, Sv2


def nice_plot(omega1, omega2, dim1, dim2, label1, label2):
    title = 'Plot in the ' + label1 + ' --' + label2 + ' domains'
    dim1 -= 1  # Adjust for vector index.
    dim2 -= 1  # Adjust for vector index.
    plt.plot(omega1[dim1, :], omega1[dim2, :], 'y.', label="Class One")
    plt.plot(omega2[dim1, :], omega2[dim2, :], 'g.', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()


def plot_ML_mean(vec1, mean1, vec2, mean2, dim, pts=200):
    x = range(0, pts)
    y1 = []
    y2 = []
    for i in range(pts):
        temp1 = 0
        temp2 = 0
        for j in range(i):
            temp1 += vec1[j]
            temp2 += vec2[j]
        y1.append(abs((temp1 / (i + 1)) - mean1))
        y2.append(abs((temp2 / (i + 1)) - mean2))

    title = 'Plot of the converge of M in the ' + dim + ' dimension.'
    plt.plot(x, y1, 'yo', label="Class One")
    plt.plot(x, y2, 'go', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()


def plot_ML_sigma(vec1, mean1, sigma1, vec2, mean2, sigma2, dim, pts=200):
    x = range(0, pts)
    y1 = []
    y2 = []
    for i in range(pts):
        temp1 = 0
        temp2 = 0
        y1.append(abs(maximum_likelihood_Sigma(vec1, mean1, pts=(i + 1)) - sigma1))
        y2.append(abs(maximum_likelihood_Sigma(vec2, mean2, pts=(i + 1)) - sigma2))

    title = 'Plot of the converge of Sigma in ML in the' + dim + ' domain.'
    plt.plot(x, y1, 'yo', label="Class One")
    plt.plot(x, y2, 'go', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()


def plot_ML_mean_vec(vec1, mean1, vec2, mean2, dim, pts=200):
    x = range(0, pts)
    y1 = []
    y2 = []
    for i in range(pts):
        temp1 = 0
        temp2 = 0
        y1.append(abs(maximum_likelihood_M(vec1, pts=(i + 1)) - mean1))
        y2.append(abs(maximum_likelihood_M(vec2, pts=(i + 1)) - mean2))

    title = 'Plot of the converge of The mean in ML in the' + dim + ' domain.'
    plt.plot(x, y1, 'yo', label="Class One")
    plt.plot(x, y2, 'go', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()


def maximum_likelihood_M(vec, pts=200):
    '''Assume that vec comes in as a numpy vector. This vector is then summed
       and divided by the length of itself to produce a vector of means.'''
    # Must return the means of the vector.
    mean = np.transpose(np.array([np.sum(vec, axis=1)])) / pts
    return mean


def maximum_likelihood_Sigma(vec, M, pts=200):
    '''Assume that vec and M come in as a numpy vector. Using the vector and
       the mean (M) the covariance matrix is generated.'''
    # Given the vector and the mean (possibly just the mean) find the
    # covariance matrix and return it.
    Sigma = ((vec - M) @ np.transpose(vec - M)) / pts
    return Sigma


def gaussian_quad_loss_fn(vec, Sigma, pts=200, dimensions=3):
    m_n = np.array([[1], [1], [1]])  # This is m_0
    ninv = 1 / pts
    Sigma_0 = np.identity(3)
    statement = la.inv((ninv * Sigma) + Sigma_0)
    printit = np.array([(ninv * np.sum(vec, axis=1))]).transpose()
    m_n = (ninv * (Sigma @ statement) @ m_n) + Sigma_0 @ statement @ printit
    return m_n


def bayesian_learner(vec, Sigma, pts=200, dimensions=3):
    '''Given a vector comlabel1pute the gaussian loss to generate the mean.'''
    # gaussian loss
    mean = gaussian_quad_loss_fn(vec, Sigma, pts, dimensions)
    return mean


def plot_bayes_mean(vec1, mean1, sigma1, vec2, mean2, sigma2, dim, pts=200):
    x = range(0, pts)
    y1 = []
    y2 = []
    for i in range(pts):
        temp1 = 0
        temp2 = 0
        y1.append(abs(gaussian_quad_loss_fn(vec1, sigma1, pts=(i + 1)) - mean1))
        y2.append(abs(gaussian_quad_loss_fn(vec2, sigma2, pts=(i + 1)) - mean2))
    y1 = np.array(y1)
    y2 = np.array(y2)
    title = 'Plot of the convergence of Bayesian mean in the ' + dim + ' dimension.'
    plt.plot(x, y1[:, 0, 0], 'yo', label="Class One")
    plt.plot(x, y2[:, 0, 0], 'go', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()


def parzen_window(x_i, var=0.5, pts=200, granularity=10000, dimensions=3):
    '''Given a vector, run the Parzen window function with a Gaussian
       kernel.'''
    # Each dimension
    minimum = 0
    maximum = 0
    pdf_est = []
    fx = 0
    var1 = 1 / sqrt(2 * pi * var)
    var2 = 2 * var
    minimum = np.min(x_i)  # Get something smaller and something larger so that I am well outside of the actual range
    maximum = np.max(x_i)
    x = np.linspace(minimum, maximum, num=granularity)
    for i in range(granularity):
        fx = 0
        for j in range(pts):
            fx += exp(-(pow((x[i] - x_i[j]), 2) / var2))
        fx = (fx * var1) / pts
        pdf_est.append(fx)
        fx = 0
    pdfarr = np.array(pdf_est)
    mean = sum(pdf_est) / pts
    samp_var = (pdfarr - mean) @ np.transpose(pdfarr - mean) / granularity
    return pdf_est, x, mean, samp_var


def plot_parzen_window_1(pdf1, x, dim):
    title = 'Plot of the PDF estimated with Parzen Windows in the ' + dim + ' dimension.'
    plt.plot(x, pdf1, 'y.', label="Class One")
    # plt.plot(pdf2, 'g.', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()


def plot_parzen_window_2(pdf1, x1, pdf2, x2, dim):
    title = 'Plot of the PDF estimated with Parzen Windows in the ' + dim + ' dimension.'
    plt.plot(x1, pdf1, 'y.', label="Class One")
    plt.plot(x2, pdf2, 'g.', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(dim)
    plt.ylabel('Probability')
    plt.show()
