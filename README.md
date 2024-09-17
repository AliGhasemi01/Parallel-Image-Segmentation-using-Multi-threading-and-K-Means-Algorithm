# Parallel-Image-Segmentation-using-Multi-threading-and-K-Means-Algorithm

This project is an implementation of an image segmentation method based on the paper titled **"Parallel Image Segmentation using Multi-threading and K-Means Algorithm"**. The segmentation is achieved by leveraging parallel computing (multi-threading) alongside the popular K-Means clustering algorithm, making the segmentation process more efficient and faster, especially for large images.

Link to the original paper:  
[Parallel Image Segmentation using Multi-threading and K-Means Algorithm](https://www.researchgate.net/publication/267329073_Parallel_image_segmentation_using_multi-threading_and_k-means_algorithm)

## Paper Overview

The method proposed in the paper introduces a multi-threaded approach to image segmentation using the K-Means clustering algorithm. The idea is to divide the input image into several parts (tiles), and each part is processed in parallel by a separate thread using K-Means. After processing each part independently, the results are combined to produce the final segmented image.

### Key Steps of the Proposed Method (from the paper):
1. **Image Partitioning**: 
   - The color image is divided into 2, 4, or 6 parts, and each part is assigned to a separate thread.
   
2. **Tile Segmentation**:
   - For each tile (part of the image), the height, width, and number of clusters are set, and K-Means clustering is applied to segment the pixels in that part.
   
3. **Parallel Processing**:
   - The clustering process is carried out in parallel for each image part, meaning multiple parts of the image are segmented simultaneously by different threads.
   
4. **Cluster Assignment**:
   - For each tile, pixels are reassigned to the nearest cluster centroid. If a pixel is not closest to its current centroid, it is moved to the cluster with the nearest centroid.
   
5. **Centroid Update**:
   - The centroids (centers of the clusters) are updated iteratively after each reallocation of pixels, and this process is repeated in parallel for all tiles.

6. **Image Reconstruction**:
   - After all parts are processed and segmented, the threads are joined, and the segmented tiles are merged to reconstruct the full segmented image.

## Project Overview

This repository provides a Python implementation of the method described above using OpenCV, NumPy, and multi-threading (`concurrent.futures`). The code processes an image using K-Means clustering and divides the image into parts for parallel processing, speeding up the segmentation process.

### Steps of the Code:
1. **Load the Image**: The image is loaded from the provided path using OpenCV.
   
2. **Image Partitioning**: The image is split into multiple horizontal parts (tiles) based on the number of threads.
   
3. **Parallel Processing with K-Means**: Each tile is processed by a separate thread using the K-Means algorithm to segment the image.
   
4. **Merging Segmented Parts**: After each thread has completed its work, the segmented parts are merged to form the final segmented image.

5. **Displaying & Saving the Image**: The original and segmented images are displayed side by side using Matplotlib, and the segmented image is saved to the same directory as the input image.

## How to Use

### Cloning the Repository
`
git clone https://github.com/your-username/parallel-kmeans-image-segmentation.git
cd parallel-kmeans-image-segmentation
`

### Running the Code
1. Install the required libraries:
   ```bash
   pip install opencv-python numpy matplotlib
   ```

2. Ensure you have an image (e.g., `dog-pic.jpg`) in the same directory, or provide a path to any image you want to segment.

3. Run the Jupyter Notebook or Python script that performs the image segmentation. For example, in Jupyter Notebook, run the following command:

```python
parallel_kmeans('dog-pic.jpg', k=3, num_parts=4, output_file='segmented_dog.jpg')
```

4. After running, the script will:
   - Display the original and segmented images side by side.
   - Save the segmented image (e.g., `segmented_dog.jpg`) in the same directory.
