import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, permutations


class FilterManager():
    def __init__(self, img, ee, is_grayscale=False):
        self.ee = ee
        self.is_grayscale = is_grayscale
        if not self.is_grayscale:
            self.img = self.apply_threshold(img) 
        else:
            self.img = img
        self.img_result = np.copy(img)

    def show(self, img, show_grid=True):
        """Plot the given image"""
        w, h = img.shape
        ax = plt.gca()
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ax.set_xticks(np.arange(-.5, w, 1))
        ax.set_yticks(np.arange(-.5, h, 1))
        ax.set_xticklabels(np.arange(0, w + 1, 1))
        ax.set_yticklabels(np.arange(0, h + 1, 1))
        plt.grid(show_grid)
        return plt.imshow(img, cmap=plt.cm.binary)

    def apply_threshold(self, img, threshold=128):
        """Applies the given threshold to a grayscale image, converting it into black and white"""
        result = np.copy(img)
        result[img > threshold] = 255
        result[img <= threshold] = 0
        return result

    def add_padding(self, img):
        """Adds padding to a given image and returns a new 'expanded' image"""
        radius = self.ee.shape[0] #only for square ee
        w, h = img.shape
        #new_img = np.zeros((w+2, h+2), dtype=np.uint8)
        #new_img[1:(w+1), 1:(h+1)] = np.copy(img)
        new_img = np.zeros((w + radius - 1, h + radius - 1), dtype=np.uint8)
        new_img[1:(w + radius-2), 1:(h + radius-2)] = np.copy(img)
        return new_img

    def _get_neighbors(self, img, i, j):
        """Gets the pixel(i,j) neighbors for a given image"""
        neighbors = []
        w, h = img.shape
        ee_w, ee_h = self.ee.shape
        img_w_padding = self.add_padding(img)
        neighbors = img_w_padding[i:ee_w+i, j:ee_h+j] 
        neighbors = np.delete(neighbors.flatten(), 4) #removemos el elemento central y nos quedamos con los vecinos
        neighbors = neighbors // 255 

        return neighbors

    def apply_filter(self, operator='er', n_iterations=1, temp=None):
        """Applies a morphological operator a certain number of times (n_iterations) to an image"""
        
        if temp is None:
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
                neighbors = self._get_neighbors(self.img_result, i, j)
                if operator == 'er':
                    pixel_value = self._erosion(self.img_result[i][j], neighbors)
                if operator == 'di':
                    pixel_value = self._dilation(self.img_result[i][j], neighbors)
                aux[i][j] = pixel_value
            self.img_result = aux
            
        return self.img_result
    
    def apply_filter_recursive(self, operator='er', n_iterations=1, temp=None):
        """Applies a morphological operator a certain number of times recursively (n_iterations) to an image"""

        if temp is None:
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
        
        
        if n_iterations >= 1:
            aux = np.copy(self.img_result)
            for i, j in product(range(w), range(h)):
                neighbors = self._get_neighbors(self.img_result, i, j)
                if operator == 'er':
                    pixel_value = self._erosion(self.img_result[i][j], neighbors)
                if operator == 'di':
                    pixel_value = self._dilation(self.img_result[i][j], neighbors)
                aux[i][j] = pixel_value
            self.img_result = aux
            
            return self.apply_filter(operator, n_iterations=(n_iterations - 1), temp=self.img_result)
        else:
            return self.img_result
        
    def _erosion(self, cur_pix, neighbors):
        """Modifies the current pixel value considering its neighbors and the erosion operation rules."""
        if not self.is_grayscale:
            if cur_pix // 255 == self.ee[1][1]:
                if np.sum(neighbors) + 1 < np.sum(self.ee):
                    return 0
                else:
                    return 255
            return cur_pix
        else:
            min_nb = np.min(neighbors)
            return min_nb

    def _dilation(self, cur_pix, neighbors):
        """Modifies the current pixel value considering its neighbors and the dilation operation rules."""
        if not self.is_grayscale:
            if np.sum(neighbors) > 0:
                return 255
            return cur_pix
        else:
            max_nb = np.max(neighbors)
            return max_nb

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