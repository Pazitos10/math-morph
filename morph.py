from itertools import product
import matplotlib.pyplot as plt
import numpy as np

class FilterManager():
    """FilterManager allows the user to apply a set of filters to an image
        using the given structuring element. """
    def __init__(self, selem):
        self.selem = selem
        self.selem.astype(np.int8)
        self.selem_white_idx = np.where(selem.flatten() == 1)

    def show(self, img, show_grid=True, show_ticks=False):
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

    def _center_img(self, img):
        img = self.apply_threshold(img)
        centered_img = np.ones_like(img, dtype=np.int64)
        half_w, half_h = centered_img.shape[0]//2, centered_img.shape[1]//2
        cr_img = self._crop_zero_values(img, True)
        cr_w, cr_h = cr_img.shape
        center_y = half_h - (cr_h // 2)
        center_x = half_w - (cr_w // 2)
        centered_img[center_x: center_x + cr_w, center_y: center_y + cr_h] = cr_img
        return centered_img

    def apply_threshold(self, img, threshold=.5):
        """Applies the given threshold to a grayscale image, converting it into black and white"""
        if not np.unique(img).size <= 2:
            result = np.ones_like(img, dtype=np.int8)
            result[img > threshold] = 1
            result[img <= threshold] = 0
            return result
        else:
            return img

    def _selem_is_contained_in(self, window):
        """Returns True if ee is contained in win. Otherwise, returns False"""
        return np.equal(np.take(window, self.selem_white_idx),
                        np.take(self.selem, self.selem_white_idx)).all()

    def _crop_zero_values(self, img, inverted=False):
        """Crop zero values from the image boundaries returning a new image without empty borders"""
        width, height = img.shape
        xmin, xmax = 0, width
        ymin, ymax = 0, height
        if not inverted:
            ones_x, ones_y = np.where(img == 1) #getting x,y where pixels 1 start appearing
        else:
            ones_x, ones_y = np.where(img == 0) #getting x,y where pixels 0 start appearing

        if ones_x.size > 0:
            xmin = min(ones_x)
            xmax = max(ones_x)

        if ones_y.size > 0:
            ymin = min(ones_y)
            ymax = max(ones_y)

        return img[xmin:xmax+1, ymin:ymax+1]

    def apply_filter(self, operator='er', img=None, n_iterations=1, as_gray=False):
        """Applies a morphological operator to an image multiple times (n_iterations)"""

        if not as_gray:
            img = self.apply_threshold(img)

        w, h = img.shape
        radius = self.selem.shape[0]
        pad_img_shape = (w + radius - 1, h + radius - 1)

        if operator == 'op':
            return self._opening(img)
        if operator == 'cl':
            return self._closing(img)
        if operator == 'ig':
            return self._internal_gradient(img)
        if operator == 'eg':
            return self._external_gradient(img)
        if operator == 'mg':
            return self._morphological_gradient(img)
        if operator == 'wth':
            return self._white_top_hat(img)
        if operator == 'bth':
            return self._black_top_hat(img)

        if n_iterations >= 1:

            pad_img = np.zeros(pad_img_shape).astype(np.int8)
            pad_img[radius-2:(w + radius-2), radius-2:(h + radius-2)] = img
            img_result = np.zeros(pad_img_shape).astype(np.int8)

            for i in range(w):
                for j in range(h):
                    neighbors = pad_img[i:i+radius, j:j+radius]
                    if operator == 'er':
                        img_result[i+1, j+1] = self._erosion(neighbors, as_gray)
                    if operator == 'di':
                        img_result[i+1, j+1] = self._dilation(neighbors, as_gray)

            img_result = img_result[radius-2:w+radius-2, radius-2:h+radius-2]
            return self.apply_filter(operator, img_result, n_iterations-1, as_gray)
        else:
            return img

    def _erosion(self, neighbors, as_gray):
        """Modifies the current pixel value considering its neighbors
            and the erosion operation rules."""
        if not as_gray:
            if self._selem_is_contained_in(neighbors):
                return 1
            else:
                return 0
        else:
            return np.min(neighbors)

    def _dilation(self, neighbors, as_gray):
        """Modifies the current pixel value considering its neighbors
            and the dilation operation rules."""
        if not as_gray:
            if np.sum(neighbors) > 0:
                return 1
            else:
                return 0
        else:
            return np.max(neighbors)

    def _opening(self, img):
        """Applies the opening operation"""
        return self.apply_filter('di', self.apply_filter('er', img))

    def _closing(self, img):
        """Applies the closing operation"""
        return self.apply_filter('er', self.apply_filter('di', img))

    def _internal_gradient(self, img):
        """Applies the internal gradient operation"""
        return img - self.apply_filter('er', img)

    def _external_gradient(self, img):
        """Applies the external gradient operation"""
        return self.apply_filter('di', img) - img

    def _morphological_gradient(self, img):
        """Applies the morphological gradient operation"""
        return self.apply_filter('di', img) - self.apply_filter('er', img)

    def _white_top_hat(self, img):
        """Applies the white top-hat operation"""
        return self.apply_filter('op', img) - img

    def _black_top_hat(self, img):
        """Applies the black top-hat operation"""
        bth = self.apply_filter('cl', img) - img
        bth[bth == -1] = 1
        return bth
