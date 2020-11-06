import matplotlib.pyplot as plt
import cv2
import numpy as np

class Harris:
    """
    Harris corner detection filter
    """
    def __init__(self, img):
        self.img = img
        self.I_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
        self.I_y = cv2.Sobel(img, -1, 0, 1, ksize=3)
        self.I_xy = np.float32(self.I_y) * np.float32(self.I_x)
        self.I_xx = np.float32(self.I_x) ** 2
        self.I_yy = np.float32(self.I_y) ** 2
        self.S_xx = cv2.boxFilter(self.I_xx, -1, (3, 3))
        self.S_yy = cv2.boxFilter(self.I_yy, -1, (3, 3))
        self.S_xy = cv2.boxFilter(self.I_xy, -1, (3, 3))

    def compute_r(self, i, j, k):

        det = (self.S_xx[i][j] * self.S_yy[i][j]) - (self.S_xy[i][j]**2)
        trace = self.S_xx[i][j] + self.S_yy[i][j]
        r = det - (k * (trace**2))
        return r

    def gauss_win(self):
        self.S_xx = cv2.GaussianBlur(self.I_xx, (5, 5), 3)
        self.S_yy = cv2.GaussianBlur(self.I_yy, (5, 5), 3)
        self.S_xy = cv2.GaussianBlur(self.I_xy, (5, 5), 3)

    def lambda_1(self, i, j):

        return self.S_xx[i][j]

    def lambda_2(self, i, j):

        return self.S_yy[i][j]


if __name__ == '__main__':


    image = cv2.imread("uc2.jpg")
    clr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x = Harris(clr)
    x.gauss_win()


    l1 = []
    l2 = []

    for i in range(len(x.I_xx)):
        for j in range(len(x.I_xx[0])):
            #if x.compute_r(i, j, 0.05) > 200000000:
                #image = cv2.circle(image, (j, i), radius=0, color=(0, 0, 255), thickness=2)
            l1.append(x.lambda_1(i, j))
            l2.append(x.lambda_2(i, j))

    #cv2.imwrite("uc2_corners_gauss_5x5_sig3.jpg", image)

    plt.scatter(l1, l2, marker='.')
    plt.title('Eigenvalues')

    plt.xlabel('lambda 1')
    plt.ylabel('lambda 2')
    plt.show()
    #print(x.compute_r(50, 50, 0.05))




