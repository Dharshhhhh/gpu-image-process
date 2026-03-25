import cv2
import numpy as np
import cupy as cp
import sys
import time

def grayscale_gpu(image):
    img_gpu = cp.asarray(image)
    gray = 0.299 * img_gpu[:, :, 2] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 0]
    return cp.asnumpy(gray).astype(np.uint8)

def edge_detection_gpu(image):
    img_gpu = cp.asarray(image)

    gray = 0.299 * img_gpu[:, :, 2] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 0]

    Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = cp.zeros_like(gray)
    gy = cp.zeros_like(gray)

    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            region = gray[i-1:i+2, j-1:j+2]
            gx[i, j] = cp.sum(Kx * region)
            gy[i, j] = cp.sum(Ky * region)

    edges = cp.sqrt(gx**2 + gy**2)
    return cp.asnumpy(edges).astype(np.uint8)

def grayscale_cpu(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py input.jpg")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found. Check file path.")
        return

    # CPU
    start = time.time()
    gray_cpu = grayscale_cpu(image)
    cpu_time = time.time() - start

    # GPU
    start = time.time()
    gray_gpu = grayscale_gpu(image)
    gpu_time = time.time() - start

    edges_gpu = edge_detection_gpu(image)

    cv2.imwrite("output_gray.jpg", gray_gpu)
    cv2.imwrite("output_edge.jpg", edges_gpu)

    print(f"CPU Time: {cpu_time:.5f} sec")
    print(f"GPU Time: {gpu_time:.5f} sec")

if __name__ == "__main__":
    main()