import json
import os
from collections import Counter

import pyphen
import spacy
from wordfreq import zipf_frequency


try:
    from pycld3 import NNetLanguageIdentifier  # type: ignore

    _USE_CLD3 = True
except Exception:
    _USE_CLD3 = False
    import langid  # type: ignore


TRANSCRIPTS_DIR = "/Users/dmitriy/programming/SpeechToText/transcipts"


def read_txt_files(directory_path: str) -> dict[str, str]:
    files_data: dict[str, str] = {}
    if not os.path.isdir(directory_path):
        return files_data
    for name in os.listdir(directory_path):
        if name.lower().endswith(".txt"):
            full_path = os.path.join(directory_path, name)
            try:
                with open(full_path, encoding="utf-8") as f:
                    files_data[name] = f.read()
            except UnicodeDecodeError:
                with open(full_path, encoding="cp1251", errors="ignore") as f:
                    files_data[name] = f.read()
    return files_data


if _USE_CLD3:
    _CLD3_ID = NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
else:
    try:
        langid.set_languages(["ru", "en"])  # ограничим, если используем langid
    except Exception:
        pass
_NLP_CACHE: dict[str, "spacy.language.Language"] = {}
_HYPHEN_CACHE: dict[str, pyphen.Pyphen] = {}


def detect_lang(text: str) -> str:
    if _USE_CLD3:
        res = _CLD3_ID.FindLanguage(text or "")
        if res and res.is_reliable and res.language in {"ru", "en"}:
            return res.language
        return "en"
    else:
        try:
            lang, score = langid.classify(text or "")
            return lang if lang in {"ru", "en"} else "en"
        except Exception:
            return "en"


def get_nlp(lang: str) -> "spacy.language.Language":
    if lang not in _NLP_CACHE:
        model_name = "ru_core_news_sm" if lang == "ru" else "en_core_web_sm"
        try:
            _NLP_CACHE[lang] = spacy.load(model_name)
        except Exception:
            # Автодозагрузка модели (если не установлена)
            from spacy.cli import download

            download(model_name)
            _NLP_CACHE[lang] = spacy.load(model_name)
    return _NLP_CACHE[lang]


def get_hyphenator(lang: str) -> pyphen.Pyphen:
    if lang not in _HYPHEN_CACHE:
        dic = "ru_RU" if lang == "ru" else "en_US"
        _HYPHEN_CACHE[lang] = pyphen.Pyphen(lang=dic)
    return _HYPHEN_CACHE[lang]


RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
EN_VOWELS = set("aeiouyAEIOUY")


def count_syllables(word: str, lang: str) -> int:
    if not word:
        return 0
    hyph = get_hyphenator(lang)
    inserted = hyph.inserted(word)
    if not inserted:
        return 1
    return inserted.count("-") + 1


def avg_word_length(words: list[str]) -> float:
    if not words:
        return 0.0
    total = sum(len(w) for w in words)
    return total / len(words)


def avg_word_length_from_doc(doc: "spacy.tokens.Doc") -> float:
    words = [t for t in doc if t.is_alpha]
    if not words:
        return 0.0
    total = sum(len(t.text) for t in words)
    return total / len(words)


def avg_sentence_length_from_doc(doc: "spacy.tokens.Doc") -> float:
    lens: list[int] = []
    for s in doc.sents:
        lens.append(sum(1 for t in s if t.is_alpha))
    if not lens:
        return 0.0
    return sum(lens) / len(lens)


def foreign_terms_count_doc(doc: "spacy.tokens.Doc", global_lang: str) -> int:
    cnt = 0
    for t in doc:
        if not t.is_alpha:
            continue
        if _USE_CLD3:
            res = _CLD3_ID.FindLanguage(t.text)
            if res and res.is_reliable and res.language in {"ru", "en"}:
                if res.language != global_lang:
                    cnt += 1
        else:
            try:
                lang, score = langid.classify(t.text)
                if lang in {"ru", "en"} and lang != global_lang and score >= 0.75:
                    cnt += 1
            except Exception:
                continue
    return cnt


# Базовые словари технических терминов (расширяемые)
TECH_TERMS_EN = {
    # общие ИТ/наука
    "api",
    "sdk",
    "http",
    "https",
    "tcp",
    "udp",
    "grpc",
    "json",
    "xml",
    "yaml",
    "rest",
    "soap",
    "cpu",
    "gpu",
    "ram",
    "rom",
    "sql",
    "nosql",
    "jdbc",
    "odbc",
    "oauth",
    "jwt",
    "ssh",
    "tls",
    "ssl",
    "ml",
    "ai",
    "nlp",
    "nlu",
    "ner",
    "cv",
    "svm",
    "rnn",
    "cnn",
    "lstm",
    "bert",
    "gpt",
    "transformer",
    "dataset",
    "hyperparameter",
    "optimizer",
    "regularization",
    "regression",
    "classification",
    "clustering",
    "serialization",
    "deserialization",
    "compiler",
    "interpreter",
    "framework",
    "library",
    "package",
    "container",
    "kubernetes",
    "docker",
    "virtualization",
    "microservice",
    "distributed",
    "scalability",
}

TECH_TERMS_RU = {
    "алгоритм",
    "нейросеть",
    "нейронная",
    "модель",
    "градиент",
    "оптимизация",
    "регрессия",
    "классификация",
    "кластеризация",
    "регуляризация",
    "гиперпараметр",
    "батч",
    "эпоха",
    "метрика",
    "компилятор",
    "интерпретатор",
    "фреймворк",
    "библиотека",
    "контейнер",
    "виртуализация",
    "микросервис",
    "распределённая",
    "масштабируемость",
    "сериализация",
    "десериализация",
    "инференс",
    "граф",
    "тензор",
    "градиентный",
}


def is_technical_token_lemma(lemma: str, lang: str) -> bool:
    t = (lemma or "").lower()
    if not t:
        return False
    if lang == "ru":
        return t in TECH_TERMS_RU
    if lang == "en":
        return t in TECH_TERMS_EN
    return t in TECH_TERMS_EN or t in TECH_TERMS_RU


def technical_term_density(words: list[str], lang: str) -> float:
    if not words:
        return 0.0
    tech = sum(1 for w in words if is_technical_token_lemma(w.lemma_, lang))
    return tech / len(words)


def technical_term_density_doc(doc: "spacy.tokens.Doc", lang: str) -> float:
    words = [t for t in doc if t.is_alpha]
    if not words:
        return 0.0
    tech = sum(1 for t in words if is_technical_token_lemma(t.lemma_, lang))
    return tech / len(words)


def rare_word_ratio_doc(doc: "spacy.tokens.Doc", lang: str) -> float:
    words = [t for t in doc if t.is_alpha]
    if not words:
        return 0.0

    def is_rare(text: str) -> bool:
        z = zipf_frequency(text.lower(), lang if lang in {"en", "ru"} else "en")
        return z < 3.0

    rare = sum(1 for t in words if is_rare(t.text))
    return rare / len(words)


def subordinate_clause_ratio_doc(doc: "spacy.tokens.Doc") -> float:
    SUBORD_DEPS = {"mark", "advcl", "ccomp", "xcomp", "csubj"}
    counts: list[int] = []
    for s in doc.sents:
        c = 0
        for t in s:
            dep = t.dep_
            if dep in SUBORD_DEPS or dep.startswith("acl") or "relcl" in dep:
                c += 1
        counts.append(c)
    if not counts:
        return 0.0
    return sum(counts) / len(counts)


def flesch_reading_ease_doc(doc: "spacy.tokens.Doc", lang: str) -> float:
    sents = list(doc.sents)
    words = [t for t in doc if t.is_alpha]
    if not sents or not words:
        return 0.0
    W = len(words)
    S = len(sents)
    syllables = sum(count_syllables(t.text, lang) for t in words)
    ASL = W / S
    ASW = syllables / W
    if lang == "ru":
        return 206.835 - 1.3 * ASL - 60.1 * ASW
    return 206.835 - 1.015 * ASL - 84.6 * ASW


def compute_metrics(text: str) -> dict[str, float]:
    lang = detect_lang(text)
    nlp = get_nlp(lang)
    doc = nlp(text)
    metrics = {
        "avg_word_length": avg_word_length_from_doc(doc),
        "technical_term_density": technical_term_density_doc(doc, lang),
        "rare_word_ratio": rare_word_ratio_doc(doc, lang),
        "foreign_terms_count": float(foreign_terms_count_doc(doc, lang)),
        "avg_sentence_length": avg_sentence_length_from_doc(doc),
        "subordinate_clause_ratio": subordinate_clause_ratio_doc(doc),
        "flesch_reading_ease": flesch_reading_ease_doc(doc, lang),
    }
    return metrics


def aggregate_metrics(per_file: dict[str, dict[str, float]], token_counts: dict[str, int]) -> dict[str, float]:
    # Взвешиваем по числу слов (кроме foreign_terms_count — тоже можно усреднить по словам)
    totals: dict[str, float] = Counter()
    total_words = sum(token_counts.values()) or 1

    for fname, m in per_file.items():
        w = token_counts.get(fname, 0) or 0
        weight = w / total_words
        for k, v in m.items():
            totals[k] += v * weight

    return dict(totals)


def main() -> None:
    files = read_txt_files(TRANSCRIPTS_DIR)
    if not files:
        print(json.dumps({"error": "Нет .txt файлов в каталоге", "dir": TRANSCRIPTS_DIR}, ensure_ascii=False, indent=2))
        return

    per_file_metrics: dict[str, dict[str, float]] = {}
    token_counts: dict[str, int] = {}

    for fname, text in files.items():
        lang = detect_lang(text)
        doc = get_nlp(lang)(text)
        per_file_metrics[fname] = compute_metrics(text)
        token_counts[fname] = sum(1 for t in doc if t.is_alpha)

    overall = aggregate_metrics(per_file_metrics, token_counts)

    result = {"files": per_file_metrics, "overall": overall}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
