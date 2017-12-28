import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, permutations

class FilterManager():
    def __init__(self, img, ee, is_grayscale=False):
        self.ee = ee
        i_ee, j_ee = np.where(ee == 1)
        self.ij_ee = list(zip(i_ee, j_ee))

        self.is_grayscale = is_grayscale
        if not self.is_grayscale:            
            self.img = self.apply_threshold(img) 
            self.img = self.crop_zero_values(self.img)
        self.img_result = np.copy(self.img)

    def show(self, img, show_grid=True, show_ticks=False):
        """Plot the given image"""
        w, h = img.shape
        ax = plt.gca()
        if show_ticks:
            ax.set_xticks(np.arange(-.5, w, 1))
            ax.set_yticks(np.arange(-.5, h, 1))
            ax.set_xticklabels(np.arange(0, w + 1, 1))
            ax.set_yticklabels(np.arange(0, h + 1, 1))
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
        plt.grid(show_grid)
        return plt.imshow(img, cmap=plt.cm.binary_r)

    def apply_threshold(self, img, threshold=128):
        """Applies the given threshold to a grayscale image, converting it into black and white"""
        result = np.copy(img)
        result[result > threshold] = 255
        result[result <= threshold] = 0
        return result

    def add_padding(self, img):
        """Adds padding to a given image and returns a new 'expanded' image"""
        radius = self.ee.shape[0] #only for square ee's
        w, h = img.shape
        new_img = np.zeros((w + radius - 1, h + radius - 1), dtype=np.uint8)
        new_img[radius-2:(w + radius-2), radius-2:(h + radius-2)] = np.copy(img)
        return new_img

    def _get_neighbors(self, img, i, j):
        """Gets the pixel(i,j) neighbors for a given image"""
        neighbors = []
        w, h = img.shape
        ee_w, ee_h = self.ee.shape
        img_w_padding = self.add_padding(img)
        neighbors = img_w_padding[i:ee_w+i, j:ee_h+j]
        if self.is_grayscale:
            neighbors = np.delete(neighbors.flatten(), 4) #removemos el elemento central y nos quedamos con los vecinos
        neighbors = neighbors // 255 
        return neighbors
    
    def _ee_is_contained_in(self, win):
        """Returns True if ee is contained in win. Otherwise, returns False"""
        for i, j in self.ij_ee:
            if win[i, j] != 1:
                return False
        return True
    
    def crop_zero_values(self, img):
        """Crop zero values from the image boundaries returning a new image without empty borders"""
        w, h = img.shape
        xmin, xmax = 0, w
        ymin, ymax = 0, h
        ones_x, ones_y = np.where(img == 255) #obtenemos los x,y de donde comienzan los valores 255

        if len(ones_x) > 0:
            xmin = min(ones_x)
            xmax = max(ones_x)

        if len(ones_y) > 0:
            ymin = min(ones_y)
            ymax = max(ones_y)

        return img[xmin:xmax+1, ymin:ymax+1]
    

    def apply_filter(self, operator='er', n_iterations=1, temp=None):
        """Applies a morphological operator a certain number of times (n_iterations) to an image"""
        
        if temp is None:
            print("entre por la operacion ", operator)
            self.img_result = np.copy(self.img)
        else:
            self.img_result = temp
        w, h = self.img_result.shape
            
        if operator == 'op':
            return self._opening()
        if operator == 'cl':
            return self._closing()
        if operator == 'ig':
            return self._internal_gradient()
        if operator == 'eg':
            return self._external_gradient()
        if operator == 'mg':
            return self._morphological_gradient()
        if operator == 'wth':
            return self._white_top_hat()
        if operator == 'bth':
            return self._black_top_hat()
        
        
        for _ in range(n_iterations):
            aux = np.copy(self.img_result)
            for i, j in product(range(w), range(h)):
                if self.img_result[i, j] != 255:
                    continue
                else:
                    neighbors = self._get_neighbors(self.img_result, i, j)
                    if operator == 'er':
                        pixel_value = self._erosion(self.img_result[i, j], neighbors)
                    if operator == 'di':
                        pixel_value = self._dilation(self.img_result[i, j], neighbors)
                    aux[i, j] = pixel_value
            self.img_result = aux
        return self.img_result
        
    def _erosion(self, cur_pix, neighbors):
        """Modifies the current pixel value considering its neighbors and the erosion operation rules."""
        if not self.is_grayscale:
            return cur_pix if self._ee_is_contained_in(neighbors) else 0
        else: 
            return np.min(neighbors)

    def _dilation(self, cur_pix, neighbors):
        """Modifies the current pixel value considering its neighbors and the dilation operation rules."""
        if not self.is_grayscale:
            return 255 if np.sum(neighbors) > 0 else cur_pix
        else:
            return np.max(neighbors)

    def _opening(self):
        """Applies the opening operation"""
        return self.apply_filter('di', temp=self.apply_filter('er'))

    def _closing(self):
        """Applies the closing operation"""
        return self.apply_filter('er', temp=self.apply_filter('di'))

    def _internal_gradient(self):
        """Applies the internal gradient operation"""
        return self.apply_threshold(self.img) - self.apply_filter('er')

    def _external_gradient(self):
        """Applies the external gradient operation"""
        return self.apply_filter('di') - self.apply_threshold(self.img)

    def _morphological_gradient(self):
        """Applies the morphological gradient operation"""
        dilated_img = self.apply_filter('di')
        eroded_img = self.apply_filter('er')
        return dilated_img - eroded_img
    
    def _white_top_hat(self):
        """Applies the white top-hat operation"""
        return self.apply_filter('op') - self.img

    def _black_top_hat(self):
        """Applies the black top-hat operation"""
        return self.apply_filter('cl') - self.img