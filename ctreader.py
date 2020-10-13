import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
from skimage.transform import resize

def draw(images, columns=4):
    rows = int(np.ceil(images.shape[0] / columns))
    max_size = 20
    
    width = max(columns * 5, max_size)
    height = width * rows // columns

    plt.figure(figsize=(width, height))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(images.shape[0]):
        plt.subplot(rows,columns,i+1), plt.imshow(images[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()


def draw_masked(images, masks, columns=4):
    assert images.shape == masks.shape
    
    rows = int(np.ceil(images.shape[0] / columns))
    max_size = 20
    
    width = min(columns * 5, max_size)
    height = width * rows // columns

    fig = plt.figure(figsize=(width, height))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    
    X, Y = np.meshgrid(np.arange(masks.shape[1]), np.arange(masks.shape[2]))
    
    for i in range(images.shape[0]):
        ax = fig.add_subplot(rows,columns,i+1)
        if masks[i].sum() > 0:
            ax.contour(X, Y, masks[i], 1, colors='red', linewidths=0.5)
        ax.imshow(images[i], origin='lower', cmap='gray')
        plt.axis('off')
    
    plt.show()

def read_mhd(file):
    return sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkFloat32))


def read_nrrd(file):
    return sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkVectorUInt16))

def read_nii(file):
    return np.asarray(nib.load(file).dataobj).transpose(2, 1, 0)
