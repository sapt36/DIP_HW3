### Overview:

This Python program implements image processing functionalities using PyQt5 for graphical user interface (GUI) and OpenCV for image manipulation. The program has three main parts corresponding to three different image processing techniques, each presented in a tab:

1. **Part 2: Spatial Filtering** (applies various filtering operations with different mask types).
2. **Part 3: Edge Detection** (uses the Marr-Hildreth and Sobel edge detection techniques).
3. **Part 4: Local Enhancement** (uses local histogram statistics and local enhancement methods to enhance image contrast).

---

### Part 2: **Spatial Filtering Operations**

**Key Features:**
- **Mask types**: Box, Gaussian, and Custom masks can be applied.
- **Mask size**: The size of the filter mask can be adjusted (ranging from 3 to 21).
- **Sigma input**: For Gaussian masks, sigma can be manually adjusted or set to automatic.
- **Custom filters**: Users can create their own mask by manually inputting the values.

**How it works**:
1. The user loads an image.
2. Selects the filter type and mask size.
3. Applies the filter, and the processed image is displayed.

**Code details**:
- The function `manual_gaussian_kernel` generates a Gaussian kernel for image filtering.
- The function `manual_filter2D` performs 2D convolution using a user-specified kernel.

#### **Effect of Mask Size** on the Processed Images and Computation Time:
- **Smaller Mask Sizes (e.g., 3x3)**: These masks lead to less smoothing, retaining most of the image's fine details, but the noise reduction effect is weaker. The computation time is faster because there are fewer pixels involved in each convolution operation.
  
- **Larger Mask Sizes (e.g., 21x21)**: Larger masks introduce more smoothing, which helps remove more noise and create a more blurred image. However, larger masks increase the computational load, leading to longer processing times. With larger masks, the filter covers a larger area of the image, so each pixel computation takes more time.

---

### Part 3: **Edge Detection Methods**

**Key Features**:
- **Marr-Hildreth**: Uses the Laplacian of Gaussian (LoG) for edge detection by first smoothing the image and then detecting zero-crossings.
- **Sobel**: Computes the gradient of the image using Sobel kernels for detecting edges.

**How it works**:
1. The user loads an image.
2. Selects the edge detection method (Marr-Hildreth or Sobel).
3. For Marr-Hildreth, the user can set a zero-crossing threshold to detect the edges.
4. The result is displayed.

**Code details**:
- `marr_hildreth_edge_detection` function uses Gaussian smoothing followed by the Laplacian kernel to detect edges.
- `sobel_edge_detection` function uses Sobel kernels in x and y directions to compute the gradient magnitude.

#### **Effect of Zero-Crossing Threshold** on the Marr-Hildreth Edge Detection:
- **Low Threshold Values**: A low threshold (e.g., 1) means that even small changes in the Laplacian values will be considered edges. This can result in many edges being detected, including weak edges and noise.
  
- **High Threshold Values**: A higher threshold (e.g., 100) reduces the number of detected edges by ignoring small variations, focusing only on stronger edges. This results in a cleaner output but risks missing faint edges in the image.

---

### Part 4: **Local Enhancement Methods**

**Key Features**:
- **Local enhancement**: Adjusts the contrast of an image by enhancing low-contrast regions while preserving the overall brightness of the image.
- **Histogram equalization**: Applies manual histogram equalization to enhance image contrast globally.
- **Local histogram statistics enhancement**: Uses local statistics (mean and standard deviation) to enhance contrast in specific regions.

**How it works**:
1. The user loads an image.
2. The user selects a neighborhood region size `Sxy` and applies local enhancement.
3. The result is displayed, and the user can also apply histogram equalization or local histogram statistics enhancement.

**Code details**:
- The `local_enhancement` function processes each pixel by adjusting its intensity based on its neighborhood mean and standard deviation.
- The `manual_histogram_equalization` function calculates the cumulative distribution function (CDF) of the image's histogram and uses it to remap pixel values for contrast enhancement.
- The `local_histogram_statistics` function enhances dark regions by using local statistics like mean and standard deviation.

#### **Effect of Neighborhood Region Size (Sxy)** on Local Enhancement Image Processing Results:
- **Small Region Sizes (e.g., Sxy = 3)**: This results in localized contrast enhancement over a small area, where only small features or details are enhanced. It retains the sharpness of the image but may leave larger regions unaffected.
  
- **Large Region Sizes (e.g., Sxy = 20)**: Larger regions consider more pixels when computing local statistics, leading to more global contrast changes. This can produce a more generalized enhancement effect across the image, but the fine details may be less pronounced.

---

### Summary of Each Part:
1. **Part 2 (Spatial Filtering)**: Focuses on image smoothing and filtering operations with customizable masks. The size of the mask directly impacts both the image's appearance and the computation time.
  
2. **Part 3 (Edge Detection)**: Provides two edge detection methods, with Marr-Hildreth being sensitive to the zero-crossing threshold. Lower thresholds detect more edges, while higher thresholds focus on strong edges.

3. **Part 4 (Local Enhancement)**: Uses local statistics and histogram-based methods to enhance image contrast. The size of the neighborhood region (Sxy) influences how localized or global the enhancement effect is, with smaller sizes focusing on finer details and larger sizes affecting broader regions.

This program gives users flexibility in processing images by allowing them to interactively modify parameters like mask size, zero-crossing threshold, and neighborhood region size, helping them understand how these parameters affect the output results.
