from enum import IntEnum, StrEnum


class Subject(StrEnum):
    ENGLISH = "english"
    SCIENCE = "science"
    MATH = "math"
    VOCABULARY = "vocabulary"
    OTHER = "other"


class Purpose(StrEnum):
    REVIEW = "review"
    CONVERSATION = "conversation"
    QUIZ = "quiz"
    PRONUNCIATION = "pronunciation"


class Language(StrEnum):
    EN = "en"
    VI = "vi"
    BILINGUAL = "bilingual"


class Phase(StrEnum):
    WARM_UP = "warm_up"
    PRESENT = "present"
    PRACTICE = "practice"
    WRAP_UP = "wrap_up"


class ActivityType(StrEnum):
    CURIOSITY_SPARK = "curiosity_spark"
    WORD_ECHO = "word_echo"
    STORY_CONTEXT = "story_context"
    EXPLAIN_AND_DISCUSS = "explain_and_discuss"
    TRUE_OR_FALSE = "true_or_false"
    QUICK_QUIZ = "quick_quiz"
    ROLE_PLAY = "role_play"
    TEACH_PIKA = "teach_pika"
    MENTAL_MATH_DRILL = "mental_math_drill"
    WORD_PROBLEMS = "word_problems"
    CATEGORY_SORT = "category_sort"
    SPELLING_BEE = "spelling_bee"
    FUN_FACT_AND_STARS = "fun_fact_and_stars"
    NUMBER_WARMUP = "number_warmup"
    CONCEPT_INTRO = "concept_intro"
    SCORE_CHALLENGE = "score_challenge"
    PICTURE_DESCRIBE = "picture_describe"
    VOCAB_INTRO = "vocab_intro"
    RAPID_FIRE = "rapid_fire"
    SUMMARY_STARS = "summary_stars"


class AgentMode(StrEnum):
    LEARN_AGENT = "learn_agent"
    TALK_AGENT = "talk_agent"


class AgentBotId(IntEnum):
    LEARN_AGENT = 1902
    TALK_AGENT = 1900


class SafetyVerdict(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    NEEDS_REVIEW = "needs_review"


class PipelineStatus(StrEnum):
    SUCCESS = "success"
    PROCESSING = "processing"
    ERROR = "error"
