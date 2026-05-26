import asyncio
import aiohttp
import aiofiles
import os
import json
import logging
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .models import ColorArtwork
from ..decorators import time_decorator

logger = logging.getLogger("metetl")

def _process_artwork_worker(pixels: np.ndarray, metadata: dict, name: str,
                            index: int, painting_id: str) -> dict:
    logger.debug("[PID %d] Начало обработки %d (%s)", os.getpid(), index, painting_id)
    artwork = ColorArtwork(name, pixels, metadata)

    results = {
        "original": pixels,
        "grayscale": artwork.to_grayscale().pixels,
        "gauss": artwork.blur_gauss().pixels,
        "sobel": artwork.sobel_edges().pixels,
        "gamma": artwork.gamma_correction().pixels,
        "canny": cv2.Canny(pixels, 100, 200),
        "gray_cv": cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY),
        "gauss_cv": cv2.GaussianBlur(pixels, (3,3), 0),
    }

    logger.debug("[PID %d] Завершение обработки %d", os.getpid(), index)
    return results

class ImageProcessor:

    async def download_and_process_one(self, session: aiohttp.ClientSession, executor: ProcessPoolExecutor,
                                       index: int, painting_id: str, output_dir: str):
        meta_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{painting_id}"
        logger.info("[%d] Загрузка метаданных %s", index, painting_id)
        async with session.get(meta_url) as resp:
            data = await resp.json()
        img_url = data.get("primaryImageSmall")
        if not img_url:
            logger.warning("[%d] Нет изображения для %s", index, painting_id)
            return

        logger.info("[%d] Скачивание изображения %s", index, painting_id)
        async with session.get(img_url) as resp_img:
            img_bytes = await resp_img.read()
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            logger.error("[%d] Не удалось декодировать %s", index, painting_id)
            return

        orig_path = os.path.join(output_dir, f"{index}_{painting_id}_original.png")
        meta_path = os.path.join(output_dir, f"{index}_{painting_id}_metadata.json")
        _, buf = cv2.imencode('.png', img_array)
        async with aiofiles.open(orig_path, 'wb') as f:
            await f.write(buf.tobytes())
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("[%d] Изображение сохранено", index)

        loop = asyncio.get_running_loop()
        logger.info("[%d] Запуск обработки %s", index, painting_id)
        results = await loop.run_in_executor(
            executor, _process_artwork_worker,
            img_array, data, data.get('title', f'Painting_{painting_id}'),
            index, painting_id
        )

        for key, img in results.items():
            if key == "original":
                continue
            filename = f"{index}_{painting_id}_{key}.png"
            filepath = os.path.join(output_dir, filename)
            _, buf = cv2.imencode('.png', img)
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(buf.tobytes())
        logger.info("[%d] Полностью обработан %s", index, painting_id)

    @time_decorator
    async def process_batch(self, painting_ids: list[str], output_dir: str, num: int):
        import random
        selected = random.sample(painting_ids, min(num, len(painting_ids)))
        os.makedirs(output_dir, exist_ok=True)

        executor = ProcessPoolExecutor(max_workers=4)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.download_and_process_one(session, executor, idx, pid, output_dir)
                for idx, pid in enumerate(selected, start=1)
            ]
            await asyncio.gather(*tasks)

        executor.shutdown()
        logger.info("Пакетная обработка завершена")