import re

from ._zeroshot import ZeroShotTask

TEXT_COMPLEXITY_COLS = {"syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"}
TEXT_COMPLEXITY_WINDOW_SEC = 30
TEXT_COMPLEXITY_MATTR_WINDOW = 50
TEXT_COMPLEXITY_MIN_WORDS_FOR_DEPTH = 4

WPS_COLS = {"wps"}

VIEWER_ADDRESS_COLS = {"viewer_address"}
VIEWER_ADDRESS_PATTERN = re.compile(r"\b(ты|вы|смотри|смотрите|подпишись|подпишитесь|друзья|товарищи)\b", re.IGNORECASE)

VIEWER_ENGAGEMENT_COLS = {"viewer_engagement"}
VIEWER_ENGAGEMENT_TASK = ZeroShotTask(
    geracl_labels=["автор обращается к зрителю и создаёт чувство совместного участия", "автор рассказывает без обращения к зрителю"],
    nli_hypothesis="Автор обращается к зрителю, создавая ощущение совместного участия и сопричастности",
)
VIEWER_ENGAGEMENT_PATTERN = re.compile(
    r"(давайте|мы с вами|каждый из (нас|вас)|согласитесь|представьте|задумайтесь|обратите внимание|вспомните|как (вы думаете|вам кажется|считаете)|кто из вас|все мы|нас (всех|объединяет)|знакомо.{0,5}\?|вам (знакомо|известно|наверняка|случалось)|узнали себя|будьте честны|поднимите руку|признайтесь\b)",
    re.IGNORECASE,
)
VIEWER_ENGAGEMENT_REGEX_SCORE = 0.85

STORYTELLING_COLS = {"storytelling"}
STORYTELLING_TASK = ZeroShotTask(
    geracl_labels=["автор рассказывает реальную историю из своей жизни или личный опыт", "автор рассуждает, объясняет или описывает чужие события"],
    nli_hypothesis="Автор рассказывает реальный случай из своей жизни или личный опыт",
)
STORYTELLING_PATTERN = re.compile(
    r"(когда я|у меня|расскажу (историю|случай)|помню,? как я|однажды (я|мы|со мной)|мой личный (опыт|случай)|на собственном опыте|в моей (жизни|практике)|это произошло со мной)",
    re.IGNORECASE,
)
STORYTELLING_REGEX_SCORE = 0.85
STORYTELLING_ENSEMBLE_THRESHOLD = 0.60

TEXT_SENTIMENT_MODEL_ID = "fyaronskiy/ruRoberta-large-ru-go-emotions"
TEXT_SENTIMENT_BATCH_SIZE = 32
TEXT_SENTIMENT_MAX_LENGTH = 512
TEXT_EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
TEXT_SENTIMENT_COLS = {f"sent_{emotion_label}" for emotion_label in TEXT_EMOTION_LABELS}

TOPIC_SHARPNESS_COLS = frozenset({"topic_sharpness_0_100"})
TOPIC_SHARPNESS_WINDOW_SEGMENTS = 15
TOPIC_SHARPNESS_WINDOW_OVERLAP = 5
TOPIC_SHARPNESS_PROMPT = """\
Ты анализируешь фрагменты транскрипции YouTube-видео.

Для КАЖДОГО сегмента дай одну оценку «остроты темы» по шкале от 0 до 100:
- 0: нейтральный контент, бытовые темы, без острой социальной/политической подачи
- 30–50: лёгкая полемика, новости без жёсткой подачи
- 70–90: война, насилие, ненависть, экстремизм, тяжёлая политика, травмирующие детали
- 100: максимально интенсивная, провокационная или экстремальная подача острой темы

Учитывай только сказанное в сегменте (не додумывай контекст всего канала).

Фрагменты:
{segments}

Ответь СТРОГО: ровно одна строка на сегмент, формат
[номер] <целое 0–100>

Пример:
[1] 8
[2] 82
[3] 15"""

SPEECH_PREDICTABILITY_COL = "speech_predictability"
SPEECH_PREDICTABILITY_WINDOW_SEC = 5

SPEECH_INTELLIGIBILITY_COLS = {"speech_intelligibility", "speech_mumble_index"}
SPEECH_INTELLIGIBILITY_WPS_FAST = 4.0
SPEECH_INTELLIGIBILITY_MUMBLE_SCALE = 2.0
SPEECH_INTELLIGIBILITY_SMOOTH_WINDOW = 3
