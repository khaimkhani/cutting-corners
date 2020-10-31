import numpy
import scipy
import cv2
import random
from PIL import *

#img = cv2.imread("ex1.jpg")

#clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sobelx = cv2.Sobel(clr, -1, 1, 0, ksize=3)
#sobely = cv2.Sobel(clr, -1, 0, 1, ksize=3)

# gradient magnitude
#grad = numpy.array([[0] * len(sobelx[0])] * len(sobelx))
#for i in range(len(sobelx)):
#    for j in range(len(sobelx[0])):
#        grad[i][j] = ((sobelx[i][j] ** 2) + (sobely[i][j] ** 2))**0.5


def seam_carve(img, original):
    """
    Seam carving algorithm, must recieve gradient-magnitude image.
    :param img:
    :type img:
    :return:
    :rtype:

    """

    ran = random.randint(0, len(img[0] - 1))

    #while img[0][ran] > 60:
    #    ran = random.randint(0, len(img[0] - 1))

    min_index = ran
    #min_index = list(img[0]).index(min(img[0]))

    pixels = []

    i = 0
    while i < len(img):
        min_list = []
        index_list = []
        for j in [-1, 0, 1]:
            if min_index + j >= 0 and min_index + j < len(img[0]):
                min_list.append(img[i][min_index + j])
                index_list.append((i, min_index + j))
        pixels.append(index_list[min_list.index(min(min_list))])
        min_index = pixels[-1][-1]
        i += 1
        min_list.clear()
        index_list.clear()

    print(pixels)
    return delete_pixels(img, original, pixels)


def delete_pixels(img, original, pixels):
    """
    delete pixels from img given pixel tuple.
    :param img:
    :type img:
    :param pixels:
    :type pixels:
    :return:
    :rtype:
    """
    new_pixels = list(pixels)
    new_img = img.tolist()
    orig = original.tolist()

    print(len(new_img[0]))
    for tup in pixels:
        i = tup[0]
        j = tup[1]
        new_img[i].pop(j)
        orig[i].pop(j)
    print(len(new_img[0]))

    origi = numpy.array(orig)
    new_arr = numpy.array(new_img)

    # return original image to cut down time
    return (new_arr, origi)






#grad = grad.astype(numpy.uint8)
#cv2.imshow("grad", grad)
#cv2.waitKey(0)

if __name__ == "__main__":

    img = cv2.imread("ex1.jpg")

    clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(clr, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(clr, -1, 0, 1, ksize=3)

    # gradient magnitude
    grad = numpy.array([[0] * len(sobelx[0])] * len(sobelx))
    for i in range(len(sobelx)):
        for j in range(len(sobelx[0])):
            grad[i][j] = ((sobelx[i][j] ** 2) + (sobely[i][j] ** 2)) ** 0.5

    cupl = (grad, clr)
    for i in range(250):
        cupl = seam_carve(cupl[0], cupl[1])

    for i in range(0):
        cupl = seam_carve(cupl[0].T, cupl[1].T)

    originall = cupl[1]

    originall = originall.astype(numpy.uint8)
    cv2.imshow("seamed", originall)
    cv2.waitKey(0)




