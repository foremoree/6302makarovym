import asyncio
import aiohttp
import aiofiles
import sys
import os
import random
import csv
import json
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod

class Artwork(ABC):
    def __init__(self, name: str, pixels: np.ndarray, metadata: dict) -> None:
        self._name = name
        self._pixels = pixels
        self._metadata = metadata

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
        return f"Картина: {self._name} | Разрешение: {self._pixels.shape} | Метаданные: {self._metadata}\n"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        new_pixels = np.clip(self.pixels + other.pixels, 0, 255).astype(np.uint8)
        return self.__class__(f"{self._name} + {other._name}", new_pixels, {**self._metadata, **other._metadata})

    @abstractmethod
    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        pass

    def blur_of_gauss(self) -> 'Artwork':
        print(f"[PID {os.getpid()}] Применяю Гаусса к {self.name}")
        kernel_gause = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        new_matrix = self.apply_filter(kernel_gause)
        new_matrix = np.clip(new_matrix, 0, 255).astype(np.uint8)
        return self.__class__(f"Размытый {self.name}", new_matrix, self.metadata)

    def sobel_borders(self) -> 'Artwork':
        print(f"[PID {os.getpid()}] Применяю Собеля к {self.name}")
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = self.apply_filter(kernel_x)
        gy = self.apply_filter(kernel_y)
        g = np.sqrt(gx**2 + gy**2)
        g = np.clip(g, 0, 255).astype(np.uint8)
        return self.__class__(f"Собель {self.name}", g, self.metadata)

    def gamma(self) -> 'Artwork':
        print(f"[PID {os.getpid()}] Применяю гамму к {self.name}")
        gamma_val = 0.5
        gamma_pixels = 255.0 * (self.pixels / 255.0) ** gamma_val
        gamma_pixels = np.clip(gamma_pixels, 0, 255).astype(np.uint8)
        return self.__class__(f"Гамма {self.name}", gamma_pixels, self.metadata)

class ColorArtwork(Artwork):
    __slots__ = ('_name', '_pixels', '_metadata', 'channels')

    def __init__(self, name, pixels, metadata):
        super().__init__(name, pixels, metadata)
        self.channels = 3

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w, chanel = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        processed_layers = []
        for i in range(chanel):
            pad_img = np.pad(self.pixels[:, :, i], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            result = np.zeros_like(self.pixels[:, :, i], dtype=np.float64)
            for y in range(h):
                for x in range(w):
                    window = pad_img[y: y + k_h, x: x + k_w]
                    result[y, x] = np.sum(window * kernel)
            processed_layers.append(result)
        return np.dstack(processed_layers)

    def to_grayscale(self) -> 'BlackWhiteArtwork':
        b = self._pixels[:, :, 0] * 0.114
        g = self._pixels[:, :, 1] * 0.587
        r = self._pixels[:, :, 2] * 0.299
        gray = (b + g + r).astype(np.uint8)
        return BlackWhiteArtwork(self.name, gray, self.metadata)

class BlackWhiteArtwork(Artwork):
    __slots__ = ('_name', '_pixels', '_metadata', 'channels')

    def __init__(self, name, pixels, metadata):
        super().__init__(name, pixels, metadata)
        self.channels = 1

    def apply_filter(self, kernel: np.ndarray) -> np.ndarray:
        h, w = self.pixels.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        pad_img = np.pad(self.pixels, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(self.pixels, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                window = pad_img[y: y + k_h, x: x + k_w]
                result[y, x] = np.sum(window * kernel)
        return result

def process_artwork_worker(pixels: np.ndarray, metadata: dict, name: str,
                           index: int, painting_id: int) -> dict:
    print(f"[PID {os.getpid()}] Начинаю обработку изображения {index} (ID {painting_id})")
    artwork = ColorArtwork(name, pixels, metadata)

    grayscale_art = artwork.to_grayscale()
    gauss_art = artwork.blur_of_gauss()
    sobel_art = grayscale_art.sobel_borders()
    gamma_art = artwork.gamma()

    canny = cv2.Canny(pixels, 100, 200)
    gray_cv = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    gauss_cv = cv2.GaussianBlur(pixels, (3, 3), 0)

    results = {
        "original": pixels,
        "grayscale": grayscale_art.pixels,
        "gauss": gauss_art.pixels,
        "sobel": sobel_art.pixels,
        "gamma": gamma_art.pixels,
        "canny": canny,
        "gray_cv": gray_cv,
        "gauss_cv": gauss_cv,
    }
    print(f"[PID {os.getpid()}] Закончил обработку изображения {index}")
    return results

async def process_one_image(session: aiohttp.ClientSession, executor: ProcessPoolExecutor,
                            index: int, painting_id: int, output_dir: str):
    url_meta = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{painting_id}"
    print(f"[PID {os.getpid()}] [{index}] Downloading metadata for {painting_id} started")
    async with session.get(url_meta) as resp:
        data = await resp.json()
    img_url = data.get("primaryImageSmall")
    if not img_url:
        print(f"[PID {os.getpid()}] [{index}] No image URL for {painting_id}, skipping")
        return

    print(f"[PID {os.getpid()}] [{index}] Downloading image {painting_id} started")
    async with session.get(img_url) as resp_img:
        img_bytes = await resp_img.read()
    print(f"[PID {os.getpid()}] [{index}] Downloading image {painting_id} finished")

    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        print(f"[PID {os.getpid()}] [{index}] Failed to decode image {painting_id}")
        return

    _, orig_png = cv2.imencode('.png', img_array)
    orig_filename = f"{index}_{painting_id}_original.png"
    orig_path = os.path.join(output_dir, orig_filename)
    async with aiofiles.open(orig_path, 'wb') as f:
        await f.write(orig_png.tobytes())
    print(f"[PID {os.getpid()}] [{index}] Original image saved to {orig_filename}")

    meta_filename = f"{index}_{painting_id}_metadata.json"
    meta_path = os.path.join(output_dir, meta_filename)
    async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, indent=4, ensure_ascii=False))
    print(f"[PID {os.getpid()}] [{index}] Metadata saved to {meta_filename}")

    name = data.get('title', f"Painting_{painting_id}")
    loop = asyncio.get_running_loop()

    print(f"[PID {os.getpid()}] [{index}] Отправляю на параллельную обработку в процесс")
    results = await loop.run_in_executor(
        executor,
        process_artwork_worker,
        img_array, data, name, index, painting_id
    )
    print(f"[PID {os.getpid()}] [{index}] Получил результаты из параллельного процесса")

    for key, img in results.items():
        if key == "original":
            continue
        _, png_bytes = cv2.imencode('.png', img)
        filename = f"{index}_{painting_id}_{key}.png"
        filepath = os.path.join(output_dir, filename)
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(png_bytes.tobytes())
        print(f"[PID {os.getpid()}] [{index}] Saved {key} -> {filename}")

    print(f"[PID {os.getpid()}] [{index}] Fully processed {painting_id}")

async def main_async(num_images: int, csv_file: str = "MetObjects.csv"):
    painting_ids = []
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Classification") == "Paintings":
                painting_ids.append(row["Object ID"])

    if not painting_ids:
        print("No painting IDs found in CSV")
        return

    selected_ids = random.sample(painting_ids, min(num_images, len(painting_ids)))
    print(f"Selected {len(selected_ids)} image IDs")

    output_dir = f"processed_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    executor = ProcessPoolExecutor(max_workers=4)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, pid in enumerate(selected_ids, start=1):
            task = process_one_image(session, executor, idx, pid, output_dir)
            tasks.append(task)

        await asyncio.gather(*tasks)

    executor.shutdown()
    print("All images processed.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python artwork.py <number_of_images>")
        sys.exit(1)

    num_images = int(sys.argv[1])
    start_time = time.perf_counter()
    asyncio.run(main_async(num_images))
    elapsed = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
