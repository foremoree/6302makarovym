import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("metetl")

class Artwork(ABC):
    def __init__(self, name: str, pixels: np.ndarray, metadata: dict = None):
        self._name = name
        self._pixels = pixels
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __str__(self) -> str:
        return f"Artwork: {self._name}, shape={self._pixels.shape}"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        new_pixels = np.clip(self.pixels + other.pixels, 0, 255).astype(np.uint8)
        return self.__class__(f"{self._name}+{other._name}", new_pixels,
                              {**self._metadata, **other._metadata})

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        pass

    def blur_gauss(self) -> 'Artwork':
        logger.debug("Гауссово размытие для %s", self.name)
        kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
        new_pixels = np.clip(self.apply_filter(kernel), 0, 255).astype(np.uint8)
        return self.__class__(f"blur_{self.name}", new_pixels, self.metadata)

    def sobel_edges(self) -> 'Artwork':
        logger.debug("Выделение границ (Собель) для %s", self.name)
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        gx = self.apply_filter(kx)
        gy = self.apply_filter(ky)
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return self.__class__(f"sobel_{self.name}", magnitude, self.metadata)

    def gamma_correction(self, gamma: float = 0.5) -> 'Artwork':
        logger.debug("Гамма-коррекция для %s", self.name)
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
            result = np.zeros_like(self.pixels[:,:,i], dtype=np.float64)
            for y in range(h):
                for x in range(w):
                    region = padded[y:y+kh, x:x+kw]
                    result[y,x] = np.sum(region * kernel)
            layers.append(result)
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
        pad_h, pad_w = kh//2, kw//2
        padded = np.pad(self.pixels, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(self.pixels, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                region = padded[y:y+kh, x:x+kw]
                result[y,x] = np.sum(region * kernel)
        return result