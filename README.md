# seenx-ml

Репозиторий для извлечения признаков из YouTube-видео и обучения моделей удержания. Пайплайн: скачать данные, посчитать признаки, обучить модели, прогнать LOO-эксперименты и собрать отчеты.

## Что где лежит

```text
.
├── main.py                    # CLI: aggregate, train, predict и вспомогательные команды
├── run_pipeline.sh            # полный локальный прогон: веса, признаки, обучение, отчеты
├── run_all_experiments.sh     # seq/multimodal/VideoMAE/BERT эксперименты
├── run_loo_experiments.sh     # LOO-бенчмарк табличных моделей
├── requirements.txt           # pip-зависимости, включая CUDA PyTorch nightly
├── environment.yml            # conda-окружение
├── configs/                   # конфиги пайплайна
├── data/                      # входные видео, retention.csv и снапшоты
├── output/                    # посчитанные CSV с признаками
├── embeddings/                # кэш CLIP/VideoMAE/CLAP/text/audio эмбеддингов
├── static/weights/            # веса локальных моделей
├── src/
│   ├── aggregator.py          # сборка всех признаков в один CSV
│   ├── extractors/
│   │   ├── video/             # визуальные признаки
│   │   ├── audio/             # аудио признаки
│   │   └── text/              # текстовые признаки
│   ├── models/                # LSTM/Transformer/BERT/VideoMAE модели удержания
│   ├── analysis/              # отчеты, сравнения, кластеризация
│   ├── cutting_shots/         # поиск и нарезка бамперов
│   └── utils/                 # конфиги, кэш, выравнивание эмбеддингов
├── train/                     # обучение и эксперименты
├── tune_hp/                   # Optuna-тюнинг
├── tests/                     # тесты
└── analysis/                  # офлайн-анализ и графики
```

## Быстрый старт

Через `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download ru_core_news_sm
```

Для распознавания спикера нужно положить эталонное фото в `static/misc/speaker_face.png`. Основные веса (`TransNetV2`, `ArcFace`, `YOLO-face`, `RAFT`) скачивает `run_pipeline.sh`; веса HuggingFace, Whisper, CLIP, DeepFace и EasyOCR подтянутся при первом запуске.

## Как запускать

Минимальный путь:

```bash
python download_data.py --count 5
./run_pipeline.sh
```

Если нужно запустить шаги отдельно:

```bash
python main.py aggregate -v data/{video_id}/video.mp4 -o output/{video_id}_features.csv -c configs/local.json -r data/{video_id}/retention.csv
python main.py train --features_dir output --data_dir data --output_dir my_metrics --save_path static/weights/model.cbm
python main.py predict --model_path static/weights/model.cbm --video_path data/{video_id}/video.mp4 --config_path configs/local.json
```

Эксперименты:

```bash
./run_loo_experiments.sh
./run_all_experiments.sh
RUN_OPTUNA=1 ./run_all_experiments.sh
```

## Какие признаки считаются

Видео: цветовая температура и насыщенность, резкость, энтропия, визуальная сложность, motion/zoom, скринкаст, оверлеи, лица, speaker probability, эмоции, эстетика, saliency, object density, depth variance, scene novelty, CLIP/VideoMAE embeddings, границы сцен через ensemble `TransNetV2 + CLAP + VideoMAE + RAFT`.

Аудио: громкость и динамика, spectral flux, beat sync, speech/music/silence, source separation, prosody, laughter, SFX, CLAP embeddings и zero-shot аудио события.

Текст: WPS, viewer address, engagement, storytelling, sentiment, topic sharpness, complexity, speech predictability, surprisal, hook score, clickbait/curiosity/expectation/friction, ad segments, title/comment/chapter features.

Мультимодальные признаки: выравнивание visual/audio/text эмбеддингов, embedding drift, video intelligence, emotion fusion, MM embeddings.

## Рабочие файлы

`data/{video_id}/video.mp4` и `data/{video_id}/retention.csv` — вход.  
`output/{video_id}_features.csv` — итоговые признаки.  
`embeddings/` — кэш тяжелых эмбеддингов, его лучше не удалять без причины.  
`experiments/` и `my_metrics/` — результаты обучения, графики и отчеты.

## Проверки

```bash
python -m pytest tests/ -v
for script in run_pipeline.sh run_all_experiments.sh run_loo_experiments.sh; do bash -n "$script"; done
```
