import unittest
import numpy as np
from skimage import io
from morph import *

class TestApplyThreshold(unittest.TestCase):
    """Test the apply_threshold function with int and float values (pos/neg)"""
    def setUp(self):
        self.img_result = np.array([[1, 1, 1, 0],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [0, 1, 1, 1]], dtype=np.uint8)

    def test_apply_threshold_1(self):
        img_1 = np.array([[255, 255, 255, 0],
                          [255, 255, 255, 255],
                          [255, 255, 255, 255],
                          [0, 255, 255, 255]], dtype=np.uint64)
        thresh = apply_threshold(img_1, threshold=200)
        self.assertTrue(np.array_equal(thresh, self.img_result))

    def test_apply_threshold_2(self):
        img_2 = np.array([[1.2, 1.5643, 1.123567, 0.3123454],
                          [1.3, 1.986879, 1.12345987, 1.0],
                          [1.1293, 1.123456, -1.567452, 1.236765],
                          [0, 1.0, 1.0, 1.0]], dtype=np.float64)
        thresh = apply_threshold(img_2)
        self.assertTrue(np.array_equal(thresh, self.img_result))

class TestBinaryMorphOps(unittest.TestCase):
    """Test binary morphologycal operations"""
    def setUp(self):
        self.img = io.imread("images/smile.png", as_grey=True)

    def test_erosion(self):
        img_result = io.imread("test_images/er_square3x3_selem.png")
        er = morph_filter('er', self.img)
        self.assertTrue(np.array_equal(er*255, img_result))

    def test_dilation(self):
        img_result = io.imread("test_images/di_square3x3_selem.png", as_grey=True)
        di = morph_filter('di', self.img)
        self.assertTrue(np.array_equal(di*255, img_result))

    def test_opening(self):
        img_result = io.imread("test_images/op_square3x3_selem.png", as_grey=True)
        op = morph_filter('op', self.img)
        self.assertTrue(np.array_equal(op*255, img_result))

    def test_closing(self):
        img_result = io.imread("test_images/cl_square3x3_selem.png", as_grey=True)
        cl = morph_filter('cl', self.img)
        self.assertTrue(np.array_equal(cl*255, img_result))

    def test_internal_gradient(self):
        img_result = io.imread("test_images/ig_square3x3_selem.png", as_grey=True)
        ig = morph_filter('ig', self.img)
        self.assertTrue(np.array_equal(ig*255, img_result))

    def test_external_gradient(self):
        img_result = io.imread("test_images/eg_square3x3_selem.png", as_grey=True)
        eg = morph_filter('eg', self.img)
        self.assertTrue(np.array_equal(eg*255, img_result))

    def test_morphologycal_gradient(self):
        img_result = io.imread("test_images/mg_square3x3_selem.png", as_grey=True)
        mg = morph_filter('mg', self.img)
        self.assertTrue(np.array_equal(mg*255, img_result))

    def test_white_top_hat(self):
        img_result = io.imread("test_images/wth_square3x3_selem.png", as_grey=True)
        wth = morph_filter('wth', self.img)
        self.assertTrue(np.array_equal(wth*255, img_result))

    def test_black_top_hat(self):
        img_result = io.imread("test_images/bth_square3x3_selem.png", as_grey=True)
        bth = morph_filter('bth', self.img)
        self.assertTrue(np.array_equal(bth*255, img_result))

class TestGrayscaleMorphOps(unittest.TestCase):
    """Test grayscale morphologycal operations"""
    def setUp(self):
        self.img = io.imread("images/smile.png", as_grey=True)

    def test_erosion(self):
        img_result = io.imread("test_images/grayscale_er.png")
        er = morph_filter('er', self.img, as_gray=True)
        er = (er*255)
        er = np.abs(er)
        self.assertTrue(er.all() == img_result.all())

    def test_dilation(self):
        img_result = io.imread("test_images/grayscale_di.png", as_grey=True)
        di = morph_filter('di', self.img, as_gray=True)
        di = (di*255)
        di = np.abs(di)
        self.assertTrue(di.all() == img_result.all())

    def test_opening(self):
        img_result = io.imread("test_images/grayscale_op.png", as_grey=True)
        op = morph_filter('op', self.img, as_gray=True)
        op = (op*255)
        op = np.abs(op)
        self.assertTrue(op.all() == img_result.all())

    def test_closing(self):
        img_result = io.imread("test_images/grayscale_cl.png", as_grey=True)
        cl = morph_filter('cl', self.img, as_gray=True)
        cl = (cl*255)
        cl = np.abs(cl)
        self.assertTrue(cl.all() == img_result.all())

    def test_internal_gradient(self):
        img_result = io.imread("test_images/grayscale_ig.png", as_grey=True)
        ig = morph_filter('ig', self.img, as_gray=True)
        ig = (ig*255)
        ig = np.abs(ig)
        self.assertTrue(ig.all() == img_result.all())

    def test_external_gradient(self):
        img_result = io.imread("test_images/grayscale_eg.png", as_grey=True)
        eg = morph_filter('eg', self.img, as_gray=True)
        eg = (eg*255)
        eg = np.abs(eg)
        self.assertTrue(eg.all() == img_result.all())

    def test_morphologycal_gradient(self):
        img_result = io.imread("test_images/grayscale_mg.png", as_grey=True)
        mg = morph_filter('mg', self.img, as_gray=True)
        mg = (mg*255)
        mg = np.abs(mg)
        self.assertTrue(mg.all() == img_result.all())

    def test_white_top_hat(self):
        img_result = io.imread("test_images/grayscale_wth.png", as_grey=True)
        wth = morph_filter('wth', self.img, as_gray=True)
        wth = (wth*255)
        wth = np.abs(wth)
        self.assertTrue(wth.all() == img_result.all())

    def test_black_top_hat(self):
        img_result = io.imread("test_images/grayscale_bth.png", as_grey=True)
        bth = morph_filter('bth', self.img, as_gray=True)
        bth = (bth*255)
        bth = np.abs(bth)
        self.assertTrue(bth.all() == img_result.all())

class TestMorphOpAppliedMultipleTimes(unittest.TestCase):
    """Test applying a morphologycal operation multiple times"""
    def setUp(self):
        self.img = io.imread("images/favicon.png", as_grey=True)
        self.selem = np.ones((3, 3), dtype=np.int64)

    def test_erosionx3(self):
        img_result = io.imread("test_images/erosion_3_times.png")
        er = morph_filter('er', self.img, self.selem, 3)
        self.assertTrue(np.array_equal(er*255, img_result))

    def test_erosionx6(self):
        img_result = io.imread("test_images/erosion_6_times.png")
        er = morph_filter('er', self.img, self.selem, 6)
        self.assertTrue(np.array_equal(er*255, img_result))

class TestDifferentSelemTypes(unittest.TestCase):
    """Test different structuring element types using the erosion operation"""
    def setUp(self):
        self.img = np.array([[1, 1, 1, 0],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [0, 1, 1, 1]], dtype=np.uint8)

    def test_square_selem(self):
        selem = np.ones((3, 3), dtype=np.int64)
        img_result = np.array([[0, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]], dtype=np.uint8)
        eroded = morph_filter('er', self.img, selem, 1, False)
        self.assertTrue(np.array_equal(eroded, img_result))

    def test_disc_selem(self):
        selem = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.int64)
        img_result = np.array([[0, 0, 0, 0],
                               [0, 1, 1, 0],
                               [0, 1, 1, 0],
                               [0, 0, 0, 0]], dtype=np.uint8)
        eroded = morph_filter('er', self.img, selem, 1, False)
        self.assertTrue(np.array_equal(eroded, img_result))

    def test_horizontal_selem(self):
        selem = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]], dtype=np.int64)
        img_result = np.array([[0, 1, 0, 0],
                               [0, 1, 1, 0],
                               [0, 1, 1, 0],
                               [0, 0, 1, 0]], dtype=np.uint8)
        eroded = morph_filter('er', self.img, selem, 1, False)
        self.assertTrue(np.array_equal(eroded, img_result))

    def test_vertical_selem(self):
        selem = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.int64)
        img_result = np.array([[0, 0, 0, 0],
                               [1, 1, 1, 0],
                               [0, 1, 1, 1],
                               [0, 0, 0, 0]], dtype=np.uint8)
        eroded = morph_filter('er', self.img, selem, 1, False)
        self.assertTrue(np.array_equal(eroded, img_result))

class TestDifferentSelemSizes(unittest.TestCase):
    """Test different structuring element types using the erosion operation"""
    def setUp(self):
        self.img = io.imread("images/favicon.png", as_grey=True)

    def test_3x3_selem(self):
        selem = np.ones((3, 3), dtype=np.uint8)
        img_result = io.imread("test_images/erosion_3x3_selem.png", as_grey=True)
        eroded = morph_filter('er', self.img, selem)
        self.assertTrue((eroded*255).all() == img_result.all())

    def test_5x5_selem(self):
        selem = np.ones((5, 5), dtype=np.uint8)
        img_result = io.imread("test_images/erosion_5x5_selem.png", as_grey=True)
        eroded = morph_filter('er', self.img, selem)
        self.assertTrue((eroded*255).all() == img_result.all())

    def test_7x7_selem(self):
        selem = np.ones((7, 7), dtype=np.uint8)
        img_result = io.imread("test_images/erosion_7x7_selem.png", as_grey=True)
        eroded = morph_filter('er', self.img, selem)
        self.assertTrue((eroded*255).all() == img_result.all())

class TestMorphReconstruction(unittest.TestCase):
    """Test morphologycal reconstruction operation"""
    def setUp(self):
        self.img = io.imread("images/chars.bmp", as_grey=True)
        self.img = 1 - self.img
        self.selem = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.int64)
        self.img_result = io.imread("test_images/morph_recon.png")

    def test_morphologycal_reconstruction(self):
        mark = io.imread("test_images/morph_recon_mark.png")
        recon = morphologycal_reconstruction(mark, self.img, False, sel=self.selem)
        self.assertTrue((recon*255).all() == self.img.all())