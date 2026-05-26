import numpy as np
import cv2
import os
import time

def load_image(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w, c = original.shape
    print(f"Размер: {w}x{h}")
    return original

def to_grayscale_manual(image):
    gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return gray

def to_grayscale_library(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def convolution_manual_color(image, kernel):
    h, w, c = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    result = np.zeros_like(image)
    for channel in range(c):
        padded = np.pad(image[:, :, channel], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh, j:j+kw]
                result[i, j, channel] = np.sum(region * kernel)
    return result

def convolution_library_color(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def gaussian_smooth_manual_color(image):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    return convolution_manual_color(image, kernel)

def gaussian_smooth_library_color(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def sobel_edges_manual_color(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w, c = image.shape
    magnitude = np.zeros((h, w))
    for channel in range(c):
        ch = image[:, :, channel].astype(np.float32)
        gx = convolution_manual_color(ch.reshape(h, w, 1), sobel_x)[:, :, 0]
        gy = convolution_manual_color(ch.reshape(h, w, 1), sobel_y)[:, :, 0]
        magnitude += np.sqrt(gx ** 2 + gy ** 2)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
    return magnitude.astype(np.uint8)

def sobel_edges_library_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma=1.5):
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)

def process_and_save(image_path, original):
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    start = time.time()
    gray_manual = to_grayscale_manual(original)
    manual_time = time.time() - start
    print(f"Моя реализация grayscale: {manual_time:.4f} с")
    start = time.time()
    gray_library = to_grayscale_library(original)
    lib_time = time.time() - start
    print(f"OpenCV grayscale: {lib_time:.4f} с")
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gray_manual.jpg"), gray_manual)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gray_library.jpg"), gray_library)

    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    start = time.time()
    blur_manual = convolution_manual_color(original, kernel)
    manual_time = time.time() - start
    print(f"Моя реализация blur: {manual_time:.4f} с")
    start = time.time()
    blur_library = convolution_library_color(original, kernel)
    lib_time = time.time() - start
    print(f"OpenCV blur: {lib_time:.4f} с")
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_blur_manual.jpg"), cv2.cvtColor(blur_manual, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_blur_library.jpg"), cv2.cvtColor(blur_library, cv2.COLOR_RGB2BGR))

    start = time.time()
    gauss_manual = gaussian_smooth_manual_color(original)
    manual_time = time.time() - start
    print(f"Моя реализация Gauss: {manual_time:.4f} с")
    start = time.time()
    gauss_library = gaussian_smooth_library_color(original)
    lib_time = time.time() - start
    print(f"OpenCV Gauss: {lib_time:.4f} с")
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gauss_manual.jpg"), cv2.cvtColor(gauss_manual, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gauss_library.jpg"), cv2.cvtColor(gauss_library, cv2.COLOR_RGB2BGR))

    start = time.time()
    sobel_manual = sobel_edges_manual_color(original)
    manual_time = time.time() - start
    print(f"Моя реализация Sobel: {manual_time:.4f} с")
    start = time.time()
    sobel_library = sobel_edges_library_color(original)
    lib_time = time.time() - start
    print(f"OpenCV Sobel: {lib_time:.4f} с")
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_sobel_manual.jpg"), sobel_manual)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_sobel_library.jpg"), sobel_library)

    gamma_img = gamma_correction(original, gamma=1.5)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gamma.jpg"), cv2.cvtColor(gamma_img, cv2.COLOR_RGB2BGR))

def main():
    paintings_dir = 'paintings'
    all_files = os.listdir(paintings_dir)
    original_images = []
    for f in all_files:
        if f.endswith('.jpg') and 'painting_' in f and not any(
            suffix in f for suffix in ['_gray', '_blur', '_gauss', '_sobel', '_gamma']
        ):
            original_images.append(f)

    if not original_images:
        print("Нет оригинальных изображений для обработки.")
        return

    print("Оригинальные изображения:")
    for i, f in enumerate(original_images, 1):
        print(f"  {i}. {f}")

    choice = input("Введите номер (или Enter для последнего): ").strip()
    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(original_images):
            selected = original_images[idx]
        else:
            selected = original_images[-1]
    else:
        selected = original_images[-1]

    image_path = os.path.join(paintings_dir, selected)
    print(f"Обрабатывается: {selected}")
    original = load_image(image_path)
    if original is not None:
        process_and_save(image_path, original)

if __name__ == "__main__":
    main()
