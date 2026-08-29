from .enums import Subject, Purpose, Language

ALLOWED_SUBJECTS = {s.value for s in Subject}
ALLOWED_PURPOSES = {p.value for p in Purpose}
ALLOWED_LANGUAGES = {lang.value for lang in Language}

MAX_IMAGES = 5
MAX_IMAGE_SIZE_MB = 10
MAX_PARENT_NOTES_LENGTH = 500
