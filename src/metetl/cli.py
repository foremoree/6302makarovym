import argparse
import asyncio
import json
import logging
import sys
from .analysis.data_to_download import prepare_metadata
from .analysis.aggregations import analyze_csv
from .images.processing import ImageProcessor

logger = logging.getLogger("metetl")

def main():
    parser = argparse.ArgumentParser(prog="metetl", description="MetArt")
    subparsers = parser.add_subparsers(dest="command", help="доступные команды")

    prep = subparsers.add_parser("prepare", help="создать json со списком id картин")
    prep.add_argument("--csv", required=True, help="путь к MetObjects.csv")
    prep.add_argument("--output", required=True, help="выходной json-файл")

    proc = subparsers.add_parser("process", help="скачать и обработать изображения")
    proc.add_argument("--input", required=True, help="json-файл со списком id изображений")
    proc.add_argument("--output", default="images", help="папка для сохранения результатов")
    proc.add_argument("--num", type=int, required=True, help="количество изображений для обработки")

    anl = subparsers.add_parser("analyze", help="анализировать csv и построить графики")
    anl.add_argument("--csv", required=True, help="путь к MetObjects.csv")
    anl.add_argument("--output-dir", required=True, help="папка для сохранения графиков")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_metadata(args.csv, args.output)
    elif args.command == "process":
        with open(args.input, 'r', encoding='utf-8') as f:
            ids = json.load(f)
        if not isinstance(ids, list):
            logger.error("JSON должен содержать список ID")
            sys.exit(1)
        asyncio.run(ImageProcessor().process_batch(ids, args.output, args.num))
    elif args.command == "analyze":
        analyze_csv(args.csv, args.output_dir)
    else:
        parser.print_help()