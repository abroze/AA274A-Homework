#!/usr/bin/env python3

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """

    ########## Code starts here ##########

    if len(F.shape) == 2:
        k ,l = F.shape
        p = 1
    else:
        k, l, p = F.shape

    F_flat = F.flatten()

    row = int((k-1)/2)
    col = int((l-1)/2)

    if len(I.shape) == 2:

        m, n = I.shape
        q = 1
        G = np.zeros((m, n))
        I_pad = np.pad(I, ((row,row),(col,col)), 'constant')

        for i in range(m):
            for j in range(n):
                I_flat = I_pad[i:i+k, j:j+l].flatten()
                G[i,j] = np.dot(F_flat, I_flat)


    else:

        m, n, q = I.shape
        G = np.zeros((m, n))
        I_pad = np.pad(I, ((row,row),(col,col),(0,0)), 'constant')
    
        for i in range(m):
            for j in range(n):
                I_flat = I_pad[i:i+k, j:j+l, 0:p].flatten()
                G[i,j] = np.dot(F_flat, I_flat)

    ########## Code ends here ##########
    return G


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    
    if len(F.shape) == 2:
        k ,l = F.shape
        p = 1
    else:
        k, l, p = F.shape


    if len(I.shape) == 2:
        m, n = I.shape
        q = 1
    else:
        m, n, q = I.shape


    row = int((k-1)/2)
    col = int((l-1)/2)

    I_pad = np.pad(I, ((row,row),(col,col),(0,0)), 'constant')

    F_flat = F.flatten()

    G = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            I_flat = I_pad[i:i+k, j:j+l, 0:p].flatten()
            G[i,j] = np.dot(F_flat, I_flat) / (np.linalg.norm(F_flat) * np.linalg.norm(I_flat).flatten())

    ########## Code ends here ##########
    return G


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    #test_img = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    #test_img2 = cv2.imread('test_card.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 3, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    #filt5 = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    #filt6 = 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        #corr_img = corr(filt6, test_img2)
        stop = time.time()
        print('Correlation function runtime:', stop - start, 's')
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
