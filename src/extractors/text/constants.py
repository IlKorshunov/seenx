import re

from ._zeroshot import ZeroShotTask

RU_AD_PATTERNS = re.compile(
    r"(спонсор|промокод|промо.?код|скидк[аиу]|по ссылке в описании|рекламн|рекламодател|интегра[цт]и[яю]|партнёр|партнер|"
    r"переходи по ссылке|регистрируйся|установи)",
    re.IGNORECASE,
)

RU_AD_CTA_PATTERNS = re.compile(
    r"(осталось|поторопитесь|поспешите|успейте|торопитесь|не\s+переключайтесь|оставайтесь\s+с\s+нами|хочу\s+с\s+вами\s+поделиться|"
    r"переходите\s+по\s+ссылке|жмите\s+по\s+ссылке|кликайте\s+по\s+ссылке|бесплатн|рекламн|промо)",
    re.IGNORECASE,
)

CULTURE_WINDOW_SEC = 30
RU_CULTURE_KEYWORDS = re.compile(
    r"\b(фильм|кино|сериал|книга|роман|песня|альбом|клип"
    r"|эпоха|столети|век|война|революци"
    r"|изобрет|открыти|теори|закон|формул"
    r"|исторически|легендарн|знаменит|известн|культов)\b",
    re.IGNORECASE,
)
SPACY_MODELS = ("ru_core_news_sm", "ru_core_news_md")
NER_LABELS = {"PER", "PERSON", "ORG", "LOC", "GPE", "EVENT", "WORK_OF_ART", "FAC"}
MAX_NER_TEXT_LEN = 5000

QUESTION_WINDOW_SEC = 25.0
ADDRESS_WINDOW_SEC = 25.0
CLAIM_WINDOW_SEC = 25.0
RU_CLAIMS = re.compile(
    r"(секрет|раскрою|покажу|расскажу|узнаете|научу|объясню"
    r"|никто не знает|мало кто знает|вы не поверите"
    r"|впервые|эксклюзив|уникальн)",
    re.IGNORECASE,
)

RU_ADDRESS = re.compile(r"\b(ты|вы|друзья|ребята|смотри|подпишись|привет)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\d{2,}")
HOOK_QUESTION_W = 0.25
HOOK_ADDRESS_W = 0.15
HOOK_CLAIM_W = 0.30
HOOK_NUMBERS_W = 0.10
HOOK_DENSITY_W = 0.20

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
