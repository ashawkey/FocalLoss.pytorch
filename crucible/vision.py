import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

def imread(filename, use_float=True):
    """
    Read the image as it is. (Grayscale, RGB, RGBA)
    cv2.imread() needs to set FLAGS to negative.
    """
    img = plt.imread(filename)
    if use_float and np.max(img) > 1:
        img = img.astype(np.float64)/255
    return img
    # return cv2.imread(filename, -1)

def imsave(filename, img):
    """
    Save the image as it is.
    plt.imsave() can't save grayscale images.
    """
    cv2.imwrite(filename, img)

def plot_images(img, img2=None):
    """
    Plot at most 2 images.
    Support passing in ndarray or image path string.
    """
    fig = plt.figure(figsize=(20,10))
    if isinstance(img, str): img = imread(img)
    if isinstance(img2, str): img2 = imread(img2)
    if img2 is None:
        ax = fig.add_subplot(111)
        ax.imshow(img)
    else:
        height, width = img.shape[0], img.shape[1]
        if height < width:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        ax.imshow(img)
        ax2.imshow(img2)
    plt.show()


def _plot_point_cloud(ax, pc, axes=[0,1,2], keep_ratio=1.0, pointsize=0.05, color='k'):
    N = pc.shape[0]
    selected = np.random.choice(N, int(N*keep_ratio))
    if not isinstance(color, str):
        color = color[selected]
    ax.scatter(*(pc[selected[:,None], axes].T), s=pointsize, c=color, alpha=0.5)
    if len(axes)==3: 
        ax.view_init(50, 135)

def plot_point_cloud(pc, axes=[0,1,2], keep_ratio=1.0, pointsize=0.05):
    """
    pc: [N, f]
    axes: [0,1,2]
    """
    fig = plt.figure(figsize=(20,10))
    if len(axes) == 3:
        ax = fig.add_subplot(111, projection='3d')
    elif len(axes) == 2:
        ax = fig.add_subplot(111)
    else:
        print("Axes should be either 2 or 3")
        exit(1)
    _plot_point_cloud(ax, pc, axes, keep_ratio=keep_ratio, pointsize=pointsize)
    plt.show()

# plot a torch_geo data object (point in 3D space)
def plot_graph(data):
    pos = data.pos.detach().cpu().numpy()
    ppos = pos.reshape(2, -1, 3)
    edge_index = data.edge_index.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*(ppos[0].T), s=1.0, c="r")
    ax.scatter(*(ppos[1].T), s=1.0, c="g")

    num_edges = edge_index.shape[1]//2
    for i in range(num_edges):
        c = 'k' if data.edge_attr[i] == 1 else 'b'
        A, B = pos[edge_index[0][i]], pos[edge_index[1][i]]
        ax.plot([A[0],B[0]], [A[1],B[1]], [A[2],B[2]], c=c, lw=1.0)

    plt.show()

def plot_matrix(mat, path=None):
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(mat)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    #return ax

def view_batch(imgs, lbls, labels=['image', 'label'], stack=False):
    '''
    imgs: [D, H, W, C], the depth or batch dimension should be the first.
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title(labels[0])
    ax2.set_title(labels[1])
    """
    if init with zeros, the animation may not update? seems bug in animation.
    """
    if stack:
        lbls = np.stack((lbls, imgs, imgs), -1)
    img1 = ax1.imshow(np.random.rand(*imgs.shape[1:]))
    img2 = ax2.imshow(np.random.rand(*lbls.shape[1:]))
    def update(i):
        plt.suptitle(str(i))
        img1.set_data(imgs[i])
        img2.set_data(lbls[i])
        return img1, img2
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=10, blit=False, repeat_delay=0)
    plt.show()
