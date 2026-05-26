import numpy as np
from abc import ABC, abstractmethod
import time
from functools import wraps
import csv
import os
import json
import requests
import random
import cv2

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n[log {time.perf_counter():.4f}] Вызов {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[log {time.perf_counter():.4f}] Завершение {func.__name__}")
        return result
    return wrapper

def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Функция {func.__name__} выполнена за {elapsed:.4f} сек")
        return result
    return wrapper

class Artwork(ABC):
    def __init__(self, name: str, pixels: np.ndarray, metadata: dict = None):
        self._name = name
        self._pixels = pixels
        self._metadata = metadata or {}

    @property
    def name(self):
        return self._name

    @property
    def pixels(self):
        return self._pixels

    @property
    def metadata(self):
        return self._metadata

    def __str__(self):
        return f"Artwork: {self._name}, shape={self._pixels.shape}"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        new_pixels = np.clip(self.pixels + other.pixels, 0, 255).astype(np.uint8)
        return self.__class__(f"{self._name}+{other._name}", new_pixels,
                              {**self._metadata, **other._metadata})

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        pass

    def blur_gauss(self) -> 'Artwork':
        kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
        new_pixels = np.clip(self.apply_filter(kernel), 0, 255).astype(np.uint8)
        return self.__class__(f"blur_{self.name}", new_pixels, self.metadata)

    def sobel_edges(self) -> 'Artwork':
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        gx = self.apply_filter(kx)
        gy = self.apply_filter(ky)
        mag = np.sqrt(gx**2 + gy**2)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        return self.__class__(f"sobel_{self.name}", mag, self.metadata)

    def gamma_correction(self, gamma: float = 0.5) -> 'Artwork':
        corrected = 255.0 * (self.pixels / 255.0) ** gamma
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return self.__class__(f"gamma_{self.name}", corrected, self.metadata)

class ColorArtwork(Artwork):
    __slots__ = ('_name', '_pixels', '_metadata', 'channels')
    def __init__(self, name, pixels, metadata=None):
        super().__init__(name, pixels, metadata)
        self.channels = 3

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w, c = self.pixels.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        layers = []
        for i in range(c):
            padded = np.pad(self.pixels[:,:,i], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            res = np.zeros_like(self.pixels[:,:,i], dtype=np.float64)
            for y in range(h):
                for x in range(w):
                    region = padded[y:y+kh, x:x+kw]
                    res[y, x] = np.sum(region * kernel)
            layers.append(res)
        return np.dstack(layers)

    def to_grayscale(self) -> 'BlackWhiteArtwork':
        b = self._pixels[:,:,0] * 0.114
        g = self._pixels[:,:,1] * 0.587
        r = self._pixels[:,:,2] * 0.299
        gray = (b + g + r).astype(np.uint8)
        return BlackWhiteArtwork(self.name, gray, self.metadata)

class BlackWhiteArtwork(Artwork):
    __slots__ = ('_name', '_pixels', '_metadata', 'channels')
    def __init__(self, name, pixels, metadata=None):
        super().__init__(name, pixels, metadata)
        self.channels = 1

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w = self.pixels.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(self.pixels, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        res = np.zeros_like(self.pixels, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                region = padded[y:y+kh, x:x+kw]
                res[y, x] = np.sum(region * kernel)
        return res

class ImageProcessor:
    @time_decorator
    @log_decorator
    def download_random_image(self, csv_filename: str, output_dir: str) -> Artwork:
        ids = []
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    ids.append(row['Object ID'])

        img_url = None
        while not img_url:
            rand_id = random.choice(ids)
            url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{rand_id}"
            data = requests.get(url).json()
            img_url = data.get('primaryImageSmall')
            if not img_url:
                continue
            os.makedirs(output_dir, exist_ok=True)
            obj_dir = os.path.join(output_dir, f"object_{rand_id}")
            os.makedirs(obj_dir, exist_ok=True)
            with open(os.path.join(obj_dir, 'info.json'), 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=2, ensure_ascii=False)
            img_bytes = requests.get(img_url).content
            jpg_path = os.path.join(obj_dir, 'img.jpg')
            with open(jpg_path, 'wb') as imgf:
                imgf.write(img_bytes)

        return ColorArtwork(data.get('title', 'untitled'), cv2.imread(jpg_path), data)

    @time_decorator
    @log_decorator
    def process_artwork(self, artwork: Artwork, output_dir: str):
        obj_id = artwork.metadata.get('objectID', 'unknown')
        obj_dir = os.path.join(output_dir, f"object_{obj_id}")
        os.makedirs(obj_dir, exist_ok=True)

        cv2.imwrite(os.path.join(obj_dir, 'My_Halftone.jpg'), artwork.to_grayscale().pixels)
        cv2.imwrite(os.path.join(obj_dir, 'My_Gauss.jpg'), artwork.blur_gauss().pixels)
        cv2.imwrite(os.path.join(obj_dir, 'My_Sobel.jpg'), artwork.sobel_edges().pixels)
        cv2.imwrite(os.path.join(obj_dir, 'My_Gamma.jpg'), artwork.gamma_correction().pixels)

        cv2.imwrite(os.path.join(obj_dir, 'CV2_Canny.jpg'), cv2.Canny(artwork.pixels, 100, 200))
        cv2.imwrite(os.path.join(obj_dir, 'CV2_Halftone.jpg'), cv2.cvtColor(artwork.pixels, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(os.path.join(obj_dir, 'CV2_Gauss.jpg'), cv2.GaussianBlur(artwork.pixels, (3,3), 0))

if __name__ == "__main__":
    processor = ImageProcessor()
    art = processor.download_random_image("MetObjects.csv", "paintings")
    print(art)
    processor.process_artwork(art, "paintings")
    bright = art + art
    cv2.imwrite(f"paintings/object_{art.metadata.get('objectID')}/Added.jpg", bright.pixels)
