DEFAULT_AUDIO_SR = 22050
CLAP_AUDIO_SR = 48000
SPEECH_AUDIO_SR = 16000

FRAME_LENGTH = 2048
HOP_LENGTH = 512

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
SPEECH_EMOTION_MODEL_ID = "superb/wav2vec2-large-superb-er"
EMBEDDINGS_ROOT = "embeddings"
CHUNK_SEC = 1
ZERO_SHOT_TEMPERATURE = 0.07
SPECTRAL_EPS = 1e-8

STEMS_DIRNAME = "stems"
STEM_VOCALS = "vocals.mp3"
STEM_MIXED = "mixed.mp3"
STEM_OTHER = "other.mp3"
STEM_DRUMS = "drums.mp3"
STEM_BASS = "bass.mp3"
STEM_FILES = (STEM_VOCALS, STEM_MIXED, STEM_OTHER, STEM_DRUMS, STEM_BASS)

DEMUCS_MODEL = "htdemucs"
DEMUCS_MAX_SEGMENT = 7.8
DEMUCS_AUDIO_SR = 44100
DEMUCS_EXTENSIONS = ["mp3", "wav", "ogg", "flac"]
DEMUCS_TWO_STEMS = None
DEMUCS_MP3 = True
DEMUCS_MP3_RATE = 320
DEMUCS_FLOAT32 = False
DEMUCS_INT24 = False

SPEECH_PROMPTS = ["speech", "talking", "narration"]
MUSIC_PROMPTS = ["music playing", "background music"]
SILENCE_PROMPTS = ["silence", "quiet"]

LOUDNESS_VARIANCE_WINDOW_SEC = 10
LOUDNESS_NOVELTY_ROLLING_SEC = 10
LOUDNESS_Z_SPIKE = 2.0
LOUDNESS_Z_DROP = -2.0
LOUDNESS_MIN_PERIODS_ROLL = 3

BEAT_TOLERANCE_SEC = 1
BEAT_SYNC_WINDOW_SEC = 15

LAUGHTER_CLAP_WEIGHT = 0.80
LAUGHTER_ACOUSTIC_WEIGHT = 0.20
LAUGHTER_REGEX_POSITIVE = 1.0
LAUGHTER_PROMPTS = ["laughter", "people laughing", "giggling", "audience laughter", "chuckling"]
LAUGHTER_SILENCE_PROMPTS = ["silence", "quiet background"]
LAUGHTER_RE_PATTERN = r"|(?:ха\s*[-]?\s*){2,}|(?:хе\s*[-]?\s*){2,}|(?:хи\s*[-]?\s*){2,}"

SFX_PROMPTS = ["sound effect", "whoosh sound", "impact sound effect", "transition sound effect", "beep sound", "cartoon sound effect", "explosion sound", "buzzer"]

SPEECH_MUSIC_WINDOW_SEC = 10
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_STRETCH_MIN_SEC = 3
MUSIC_CHANGE_PERCENTILE = 95.0
