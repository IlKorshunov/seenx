"""Aggregates all YouTube comments and generates insights using Qwen.

Extracts:
- Top 10 ideas for channel development / next videos
- Top 3 negative points (what viewers dislike)
- Additional audience patterns and insights

Usage:
  python -m src.analysis.comment_insights \
    --comments-dir get_data/comments \
    --output get_data/comment_insights.md \
    --max-comments 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_LLM_MODEL_ID = "Qwen/Qwen3-4B"
_MAX_NEW_TOKENS = 3072
_MAX_CHARS_PER_CHUNK = 40_000  # ~10k-12k tokens, safe for 16GB GPU


def _load_all_comments(comments_dir: Path) -> list[dict]:
    paths = list(comments_dir.rglob("comments.json"))
    logger.info("Found %d comments.json files", len(paths))
    out: list[dict] = []
    for path in tqdm(paths, desc="Loading comments"):
        vid = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        for thread in data.get("threads", []):
            text = (thread.get("text") or "").strip()
            if text:
                out.append({"video_id": vid, "text": text, "like_count": int(thread.get("like_count", 0)), "replies_count": len(thread.get("replies", []))})
    return out


def _format_comments_for_prompt(comments: list[dict], max_comments: int) -> str:
    """Sorts comments by engagement and formats them into a string, bounded by token/char limit."""
    comments.sort(key=lambda x: x["like_count"], reverse=True)

    top_comments = comments[:max_comments]
    lines = []
    total_chars = 0

    for i, c in enumerate(top_comments, 1):
        clean_text = c["text"].replace("\n", " ")
        line = f"[{i}] [Лайков: {c['like_count']} | Видео: {c['video_id']}] {clean_text}"

        if total_chars + len(line) > _MAX_CHARS_PER_CHUNK:
            logger.info(f"Reached context limit ({_MAX_CHARS_PER_CHUNK} chars) at comment #{i}. Truncating.")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _qwen_generate(prompt: str) -> str:
    """Loads Qwen model, generates response, and unloads model."""
    logger.info(f"Loading {_LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    logger.info("Generating insights with Qwen (this may take a few minutes)...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False, temperature=1.0)

    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()

    if "</think>" in full:
        response = full.split("</think>", 1)[1].strip()
    else:
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Cleanup memory
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate comments and generate insights via Qwen")
    parser.add_argument("--comments-dir", type=Path, default=Path("get_data/comments"), help="Directory containing comments.json files")
    parser.add_argument("--output", type=Path, default=Path("get_data/comment_insights.md"), help="Path to save the generated markdown report")
    parser.add_argument("--max-comments", type=int, default=1500, help="Max top comments to include in the prompt to avoid OOM")
    args = parser.parse_args()

    if not args.comments_dir.exists():
        logger.error(f"Comments directory not found: {args.comments_dir}")
        return

    all_comments = _load_all_comments(args.comments_dir)
    logger.info(f"Loaded {len(all_comments)} top-level comments across all videos.")

    if not all_comments:
        logger.warning("No comments found. Exiting.")
        return

    comments_text = _format_comments_for_prompt(all_comments, args.max_comments)
    logger.info(f"Formatted top {args.max_comments} comments. Prompt payload size: {len(comments_text)} chars.")

    prompt = f"""\
Ты — опытный YouTube-продюсер и аналитик аудитории музыкального канала (канал называется "Лирикс" или "Лонгплей", автор — Ваня).
Ниже приведены самые залайканные и обсуждаемые комментарии со всего канала (отсортированы по популярности).

Проанализируй эти комментарии и составь подробный аналитический отчет. Отчет должен содержать:

1. **Топ-10 конкретных идей для развития канала или тем для следующих видео**, исходя из зрительских симпатий.
2. **Топ-3 негативных момента**: что зрителям не нравится в КОНТЕНТЕ, подаче, форматах или чего им не хватает (обязательно с примерами конкретных цитат из комментариев).
3. **Общие паттерны и инсайты аудитории**: настроение зрителей, типичные шутки (локальные мемы канала), частые запросы к автору.
4. **Самые успешные форматы**: какие типы видео (исходя из ID видео в комментариях) вызывают наибольший восторг и почему.
5. **Портрет аудитории**: кто эти люди, чем они увлекаются, какой у них возраст/интересы (если это можно понять по стилю общения и темам).

СТРОГИЕ ПРАВИЛА ОФОРМЛЕНИЯ И АНАЛИЗА (КРИТИЧЕСКИ ВАЖНО):
- КАТЕГОРИЧЕСКИ ИГНОРИРУЙ любые комментарии про политику, войну, национальности и страны. Сосредоточься ТОЛЬКО на музыке, контенте канала, форматах и фактах!
- В каждой из 10 идей и 3 негативных моментов должны быть УНИКАЛЬНЫЕ мысли. Запрещено повторять одни и те же идеи или цитаты в разных пунктах! 
- КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО перечислять голые номера (типа "комментарии 1, 25, 26").
- Для каждой идеи или вывода приведи МАКСИМУМ 3 самых ярких цитаты из комментариев (не больше 3!). Укажи их оригинальный текст.
- Используй Markdown для красивого оформления (жирный текст, списки, выделение цитат через > ).

Комментарии аудитории:
{comments_text}
"""

    report = _qwen_generate(prompt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    logger.info(f"Successfully saved comment insights report to {args.output}")


if __name__ == "__main__":
    main()
