import numpy
import scipy
import cv2
import random
from PIL import *

#img = cv2.imread("ex1.jpg")

#clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sobelx = cv2.Sobel(clr, -1, 1, 0, ksize=3)
#sobely = cv2.Sobel(clr, -1, 0, 1, ksize=3)




def seam(img, original, iters):
    """
    new attempt using dynamic programming
    """
    p = iters
    curr_img = img.copy()
    curr_orig = original.copy()
    new = [curr_img, curr_orig, 0]
    while p > 0:
        energy = numpy.zeros((len(new[0]), len(new[0][0])))

        for i in range(len(energy[0])):
            energy[0][i] = new[0][0][i]

        new[-1] = energy
        for j in range(1, len(new[0])):
            new[-1] = compute_energies(new[0], j, new[-1])

        new = remove_min_seam(new[0], new[1], new[2])
        print(new[-1])
        p -= 1

    return new


def remove_min_seam(img, original, energy):

    indices = []
    i = len(energy) - 1
    energy = energy.tolist()
    indices.append(energy[i].index(min(energy[i])))
    ind = indices[0]
    i -= 1
    min_list = []
    min_ind = []
    while i >= 0:
        for k in [-1, 0, 1]:
            if len(energy[0]) > k + ind >= 0:
                min_list.append(energy[i][k + ind])
                min_ind.append(k + ind)
        i -= 1
        index = min_ind[min_list.index(min(min_list))]
        ind = index
        indices.append(index)

    u = len(energy) - 1
    new_img = img.tolist()
    new_original = original.tolist()
    print(len(new_img), len(new_img[0]))
    print(len(new_original), len(new_original[0]))
    while len(indices) != 0:
        new_ind = indices.pop(0)
        new_img[u].pop(new_ind)
        new_original[u].pop(new_ind)

        energy[u].pop(new_ind)
        u -= 1

    energy = numpy.array(energy, dtype=int)
    img = numpy.array(new_img, dtype=int)
    original = numpy.array(new_original, dtype=int)

    return [img, original, energy]



def compute_energies(img, j, energy):

    #energy = energy.tolist()
    for i in range(len(img[j])):
        pixel_min = []
        for k in [-1, 0, 1]:
            if len(img[j]) > i + k >= 0:
                pixel_min.append(energy[j - 1][k + i])
        energy[j][i] = min(pixel_min) + img[j][i]
        print((j, i))

    #energy = numpy.array(energy)

    return energy









#grad = grad.astype(numpy.uint8)
#cv2.imshow("grad", grad)
#cv2.waitKey(0)

if __name__ == "__main__":

    img = cv2.imread("ex1.jpg")

    clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(clr, -1, 1, 0, ksize=5)
    sobely = cv2.Sobel(clr, -1, 0, 1, ksize=5)

    # gradient magnitude

    grad = numpy.zeros((len(sobelx), len(sobelx[0])))
    for i in range(len(sobelx)):
        for j in range(len(sobelx[0])):
            grad[i][j] = ((sobelx[i][j] ** 2) + (sobely[i][j] ** 2)) ** 0.5




    cupl = seam(grad, img, 471)
    cupl = seam(cv2.transpose(cupl[0]), cv2.transpose(cupl[1]), 0)
    #cupl = seam(cupl[0].T, cupl[1].T, 1)
    #cupl = seam(test, test, 1)


    originall = cv2.transpose(cupl[1])

    originall = originall.astype(numpy.uint8)
    cv2.imwrite("ex1carvedattempt2.jpg", originall)

