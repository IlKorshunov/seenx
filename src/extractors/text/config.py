import re


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
