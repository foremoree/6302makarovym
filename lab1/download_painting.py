import csv
import json
import random
import requests
import os

os.makedirs('paintings', exist_ok=True)

with open('MetObjects.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    paintings = []
    for row in reader:
        if row.get('Classification') == 'Paintings':
            object_id = row.get('Object ID')
            if object_id:
                paintings.append(object_id)

random_id = random.choice(paintings)
print(f"Object ID: {random_id}")

url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{random_id}"
response = requests.get(url)
painting_info = response.json()

if not painting_info.get("primaryImageSmall"):
    print("Нет ссылки на изображение", painting_info)
    exit()

image_url = painting_info['primaryImageSmall']
print(f"URL: {image_url}")

img_response = requests.get(image_url)
img_path = f'paintings/painting_{random_id}.jpg'
with open(img_path, 'wb') as img_file:
    img_file.write(img_response.content)

json_path = f'paintings/painting_{random_id}.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(painting_info, json_file, indent=2, ensure_ascii=False)

print(f"Изображение сохранено: {img_path}")
print(f"Метаданные сохранены: {json_path}")
