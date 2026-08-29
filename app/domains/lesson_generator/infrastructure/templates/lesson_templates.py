from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityTemplate:
    phase: str
    activity_type: str
    duration_min: int
    description: str


ENGLISH_TEMPLATE = [
    ActivityTemplate("warm_up", "word_echo", 2, "Pika says new word, child repeats"),
    ActivityTemplate("present", "story_context", 3, "Pika tells story using new words"),
    ActivityTemplate("practice", "quick_quiz", 3, "Ask meaning and make sentences"),
    ActivityTemplate("practice", "role_play", 3, "Role play using vocabulary"),
    ActivityTemplate("wrap_up", "summary_stars", 1, "Summarize, award stars"),
]

SCIENCE_TEMPLATE = [
    ActivityTemplate("warm_up", "curiosity_spark", 2, "Curiosity-sparking question"),
    ActivityTemplate("present", "explain_and_discuss", 4, "Explain main concept"),
    ActivityTemplate("practice", "true_or_false", 3, "True/false statements"),
    ActivityTemplate("practice", "teach_pika", 2, "Child explains back to Pika"),
    ActivityTemplate("wrap_up", "fun_fact_and_stars", 1, "Fun fact + reward"),
]

MATH_TEMPLATE = [
    ActivityTemplate("warm_up", "number_warmup", 2, "Quick mental math"),
    ActivityTemplate("present", "concept_intro", 2, "Introduce operation"),
    ActivityTemplate("practice", "mental_math_drill", 4, "Progressive difficulty drills"),
    ActivityTemplate("practice", "word_problems", 3, "Word problems"),
    ActivityTemplate("wrap_up", "score_challenge", 1, "Summary + challenge"),
]

VOCABULARY_TEMPLATE = [
    ActivityTemplate("warm_up", "picture_describe", 2, "Describe, child guesses name"),
    ActivityTemplate("present", "vocab_intro", 3, "Introduce 5-8 new words"),
    ActivityTemplate("practice", "category_sort", 3, "Sort words into groups"),
    ActivityTemplate("practice", "spelling_bee", 2, "Spell words"),
    ActivityTemplate("wrap_up", "rapid_fire", 2, "Quick reading + reward"),
]

TEMPLATE_REGISTRY: dict[str, list[ActivityTemplate]] = {
    "english": ENGLISH_TEMPLATE,
    "science": SCIENCE_TEMPLATE,
    "math": MATH_TEMPLATE,
    "vocabulary": VOCABULARY_TEMPLATE,
}


def get_template(subject: str) -> list[ActivityTemplate]:
    return TEMPLATE_REGISTRY.get(subject, VOCABULARY_TEMPLATE)
