import csv
import json
import logging

logger = logging.getLogger("metetl")

def prepare_metadata(csv_path: str, output_json: str):
    ids = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Classification') == 'Paintings' and row.get('Object ID'):
                ids.append(row['Object ID'])
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ids, f, indent=2)
    logger.info("Подготовлено %d ID изображений, сохранено в %s", len(ids), output_json)