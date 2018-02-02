""" morph.py allows the user to apply a set of filters to an image
    using the given structuring element.
    author: @pazitos10
"""
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from skimage.morphology import binary_dilation, dilation

_WIDTH = 0
_HEIGHT = 0
_SELEM = np.ones((3, 3), dtype=np.int64)

def show(img, show_grid=True, show_ticks=False):
    """Plot the given image"""
    width, height = img.shape
    axes = plt.gca()
    if show_ticks:
        axes.set_xticks(np.arange(-.5, width, 1))
        axes.set_yticks(np.arange(-.5, height, 1))
        axes.set_xticklabels(np.arange(0, width + 1, 1))
        axes.set_yticklabels(np.arange(0, height + 1, 1))
    else:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    plt.grid(show_grid)
    return plt.imshow(img, cmap=plt.cm.binary_r)

def _center_img(img):
    """Center the image"""
    img = apply_threshold(img)
    centered_img = np.ones_like(img, dtype=np.int64)
    half_w, half_h = centered_img.shape[0]//2, centered_img.shape[1]//2
    cr_img = crop_zero_values(img, True)
    cr_w, cr_h = cr_img.shape
    center_y = half_h - (cr_h // 2)
    center_x = half_w - (cr_w // 2)
    centered_img[center_x: center_x + cr_w, center_y: center_y + cr_h] = cr_img
    return centered_img

def apply_threshold(img, threshold=.5):
    """Applies the given threshold to an image, converting it into black and white"""
    result = np.ones_like(img, dtype=np.int64)
    result[np.abs(img) <= threshold] = 0
    result[np.abs(img) > threshold] = 1
    return result

def _selem_is_contained_in(window):
    """Returns True if ee is contained in win. Otherwise, returns False"""        
    #selem_white_idx = np.where(_SELEM.flatten() == 1)[0]
    #selem_sum = np.sum(_SELEM)
    #win_sum = np.sum(np.take(window, selem_white_idx))
    #return True if win_sum == selem_sum else False
    
    idx_selem = np.where(_SELEM.ravel() == 1)[0]
    idx_win = np.where(window.ravel() == 1)[0]
    return set(idx_selem) <= set(idx_win)
    
    
def crop_zero_values(img, inverted=False):
    """Crop zero values from the image boundaries returning a new image without empty borders"""
    width, height = img.shape
    xmin, xmax = 0, width
    ymin, ymax = 0, height
    ones_x, ones_y = np.where(img == 1) if not inverted else np.where(img == 0)
    if ones_x.size > 0:
        xmin, xmax = min(ones_x), max(ones_x)
    if ones_y.size > 0:
        ymin, ymax = min(ones_y), max(ones_y)
    return img[xmin:xmax+1, ymin:ymax+1]

def add_padding(img, radius):
    width, height = img.shape
    pad_img_shape = (width + radius - 1, height + radius - 1)
    pad_img = np.zeros(pad_img_shape).astype(np.float64)
    pad_img[radius-2:(width + radius-2), radius-2:(height + radius-2)] = img
    return pad_img

def process_pixel(i, j, operation, as_gray, img):
    radius = _SELEM.shape[0]
    neighbors = img[i:i+radius, j:j+radius]
    if as_gray:
        neighbors = np.delete(neighbors.flatten(), radius+1)
    return operation(neighbors, as_gray)


def _apply_filter(operation, img, as_gray, n_iterations, sel):
    """Applies a morphological operator a certain number of times (n_iterations) to an image"""
    global _SELEM, _WIDTH, _HEIGHT
    _SELEM = sel
    orig_width, orig_height = _WIDTH, _HEIGHT
    img = img if as_gray else apply_threshold(img)
    radius = _SELEM.shape[0]
    width, height = img.shape
    prod = product(range(width), range(height))
    pad_img = add_padding(img, radius)
    img_result = np.zeros_like(img)
    if n_iterations >= 1:
        for i, j in prod:
            if operation == 'er' and pad_img[i, j] == 1 :
                img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
            else:
                img_result[i, j] = process_pixel(i, j, operation, as_gray, pad_img)
        return _apply_filter(operation, img_result, as_gray, n_iterations-1, sel)
    return img

def _erosion(img, as_gray, n_iterations, sel):
    """Interface function to call erosion filter"""
    return _apply_filter(_apply_erosion, img, as_gray, n_iterations, sel)

def _dilation(img, as_gray, n_iterations, sel):
    """Interface function to call erosion filter"""
    return _apply_filter(_apply_dilation, img, as_gray, n_iterations, sel)

def _apply_erosion(neighbors, as_gray):
    """Modifies the current pixel value considering its neighbors
        and the erosion operation rules."""
    if not as_gray:
        if max(neighbors.ravel()) == 1:
            if _selem_is_contained_in(neighbors):
                return 1
        return 0
    return min(neighbors.ravel())

def _apply_dilation(neighbors, as_gray):
    """Modifies the current pixel value considering its neighbors
        and the dilation operation rules."""
    if not as_gray:
        if max(neighbors.ravel()) == 0:
            return 0
        return 1
    return max(neighbors.ravel())

def _opening(img, as_gray, n_iterations, sel):
    """Applies the opening operation"""
    eroded = _erosion(img, as_gray, n_iterations, sel)
    return _dilation(eroded, as_gray, n_iterations, sel)

def _closing(img, as_gray, n_iterations, sel):
    """Applies the closing operation"""
    dilated = _dilation(img, as_gray, n_iterations, sel)
    return _erosion(dilated, as_gray, n_iterations, sel)

def _internal_gradient(img, as_gray, n_iterations, sel):
    """Applies the internal gradient operation"""
    img = img if as_gray else apply_threshold(img)
    return img - _erosion(img, as_gray, n_iterations, sel)

def _external_gradient(img, as_gray, n_iterations, sel):
    """Applies the external gradient operation"""
    img = img if as_gray else apply_threshold(img)
    return _dilation(img, as_gray, n_iterations, sel) - img

def _morphologycal_gradient(img, as_gray, n_iterations, sel):
    """Applies the morphologycal gradient operation"""
    dilated = _dilation(img, as_gray, n_iterations, sel)
    eroded = _erosion(img, as_gray, n_iterations, sel)
    return dilated - eroded

def _white_top_hat(img, as_gray, n_iterations, sel):
    """Applies the white top-hat operation"""
    if not as_gray:
        img = apply_threshold(img)
        wth = np.abs(_opening(img, as_gray, n_iterations, sel) - img)
        return apply_threshold(wth)
    return _opening(img, as_gray, n_iterations, sel) - img

def _black_top_hat(img, as_gray, n_iterations, sel):
    """Applies the black top-hat operation"""
    if not as_gray:
        img = apply_threshold(img)
        bth = np.abs(_closing(img, as_gray, n_iterations, sel) - img)
        return apply_threshold(bth)
    return _closing(img, as_gray, n_iterations, sel) - img

def morphologycal_reconstruction(mark, mask, as_gray, sel):
    """Reconstructs objects in an image based on a mark and a mask (original image)"""
    global _WIDTH, _HEIGHT
    _WIDTH, _HEIGHT = mask.shape
    mask = mask if as_gray else apply_threshold(mask)
    done = False
    prev_reconst = np.zeros_like(mark)
    aux = mark
    while not done:
        reconst = np.logical_and(aux, mask)
        aux = _dilation(reconst, as_gray, 1, sel=sel)
        if not np.array_equal(reconst, prev_reconst):
            prev_reconst = reconst
        else:
            done = True
    return 1 - reconst

_OPS = {
    'er': _erosion,
    'di': _dilation,
    'op': _opening,
    'cl': _closing,
    'ig': _internal_gradient,
    'eg': _external_gradient,
    'mg': _morphologycal_gradient,
    'wth': _white_top_hat,
    'bth': _black_top_hat
}

def morph_filter(operator='er',
                 img=None,
                 sel=np.ones((3, 3), dtype=np.int64),
                 n_iterations=1,
                 as_gray=False):
    """Allows to apply multiple morphologycal operations over an image"""
    global _WIDTH, _HEIGHT
    _WIDTH, _HEIGHT = img.shape
    return _OPS[operator](img, as_gray, n_iterations, sel)
