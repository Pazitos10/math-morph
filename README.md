### Mathematical Morphology Operations done with Python.

##### How to use

Suppose we have to erode this dummy binary image:

```python
import numpy as np
import matplotlib.pyplot as plt
from morph import morph_filter, show
​
img = np.array([[1,1,1,0],
                [1,1,1,1], 
                [1,1,1,1],
                [0,1,1,1]], dtype=np.uint8)
show(img, show_ticks=True, show_grid=True)
```
![](images/dummy-img.png)

We need to create a structuring element (in this case a 3x3, square) in order to erode it and then call `morph_filter()` as follows:

```python
selem = np.ones((3, 3), dtype=np.int64)
eroded_img = morph_filter('er', img, as_gray=False, n_iterations=1, sel=selem)
show(eroded_img, show_ticks=True, show_grid=True)
```
And this is the result:
![](images/dummy-img-eroded.png)

##### Public functions

| Name | Description | Options |
| --- | --- | --- | 
| `morph_filter(operator='er', img=None, as_gray=False, n_iterations=1, sel=selem)`| Applies a morphological operator to an image multiple times (n_iterations) | operator: <ul><li>`'er'`: Erosion. </li><li>`'di'`: Dilation. </li><li>`'op'`: Opening. </li><li>`'cl'`: Closing. </li><li>`'ig'`: Internal gradient. </li><li>`'eg'`: External Gradient. </li><li>`'mg'`: Morphological gradient. </li><li>`'wth'`: White top-hat. </li><li>`'bth'`: Black top-hat. </li></ul> img: <ul><li> Any numpy array with values between 0 and 1, representing a grayscale image. (default=`None`)</li></ul> n_iterations: <ul><li> Integer number indicating how many times to apply the selected filter over an image. (default=`1`)</li></ul> as_gray: <ul><li> `True` or `False` indicating whether to proceed executing grayscale or binary operations (default=`False`). </li></ul> sel: <ul><li>bidimensional numpy array representing the structuring element. (default=`np.ones((3,3), dtype=np.int64)`)</li><ul>
| `apply_threshold(img, threshold=.5)` | Applies the given threshold to a grayscale image, converting it into black and white (0,1) | threshold: <ul><li> Float number between 0.0 and 1.0. (default=`.5`) </li></ul>
| `show(img, show_grid=True, show_ticks=False)` | Plot the given image (shorthand for plt.imshow with preset parameters) | 
| `morphological_reconstruction(mark, mask, as_gray=False, sel=selem)` | Reconstructs objects in an image based on a mark and a mask (original image) | as_gray: <ul><li> `True` or `False` indicating whether to proceed executing grayscale or binary operations (default=`False`). </li></ul> **Note**: `mark` and `mask` must have the same dimensions. 


##### Examples

Structuring elements:
![](images/selem-types.png)

Different filters applied to the original image:
![](images/filters.png)

A single filter applied multiple times to the original image:
![](images/filters-multiple-times.png)

Different structuring element types (3x3)
![](images/filters-multiple-selem.png)

Different structuring element sizes
![](images/filters-multiple-selem-sizes.png)

Applying grayscale filters
![](images/filters-grayscale.png)

Morphological reconstruction

Original:

![](images/chars.bmp)

Applying erosion filter multiple times with an appropiated structuring element, we create a mark/seed to isolate 2 and Q.

![](images/morphological-mark.png)

Then we use `morphological_reconstruction(mark, mask)` and here is the result:

![](images/morphological-reconstruction.png)

##### Requirements

In order to use morph.py you need to run the following command:

```pip install matplotlib numpy scikit-image```

##### Tests

There are 30 tests written using `unittest` and `skimage` (only to read the test images). 
To run them execute the following command:  `python -m unittest test_morph.py`


##### To Do

Fix performance issues if possible. Currently, skimage morphology operators are faster, even with big images. For example, using morph.py for 512x512 images, the erosion filter takes ~6s vs. ~100ms (skimage).
