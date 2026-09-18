"""Rule-based question answering over a timestamped transcript.

No neural model runs in this half of the pipeline, on purpose. The dataset's
own description of a ``hard_negative`` is "the right drug at the wrong dose,
the right course at the wrong length" — a near-miss that is topically
identical to the true statement and differs only in one checkable detail (a
number, a unit, a swapped word). Judging *topical* similarity, which is what
an embedding model is good at, gets every one of those wrong by construction.
What separates them is whether the transcript's own words actually back up
the question's specific claim, so that is what this module checks directly:

1. Find the passage (a short run of consecutive Whisper segments) that is
   lexically closest to the question, via TF-IDF cosine similarity. This also
   answers ``off_topic`` questions almost for free — nothing in the
   conversation is close to them.
2. Pull out any (number, unit) pair the question asserts (a dose, a
   duration, a percentage, ...) and check it against the numbers actually
   present in that passage. A different number under the same unit is the
   clearest possible signal of a hard negative, so it overrides everything
   else.
3. Otherwise fall back to how much of the question's remaining content words
   are literally present in the passage. High coverage plus a topical match
   is a "yes"; a topical match whose specific words are not there is a "no".

Every threshold here (``ANSWER_SIMILARITY_THRESHOLD``, ``ANSWER_COVERAGE_THRESHOLD``)
is a guess tuned by inspection, not against a scored run, because doing that
needs a real transcription of the training set. Retune them with
``local_evaluator.py`` wherever ASR is actually available.
"""

import difflib
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import Span

logger = logging.getLogger(__name__)

# How lexically close the best-matching passage has to be before a question is
# even considered on-topic. Below this, the answer is "no" and there is
# nothing worth pointing at — this is what catches ``off_topic`` questions.
SIMILARITY_THRESHOLD = float(os.environ.get('ANSWER_SIMILARITY_THRESHOLD', '0.10'))

# Fraction of the question's own content words that have to show up in the
# matched passage before an otherwise-unverified "yes" is allowed through.
COVERAGE_THRESHOLD = float(os.environ.get('ANSWER_COVERAGE_THRESHOLD', '0.55'))

# How many consecutive Whisper segments to try merging into one candidate
# evidence span. Some annotated spans are one short sentence; others run
# across several, so a fixed window size would fit neither well.
MAX_WINDOW_SEGMENTS = int(os.environ.get('ANSWER_MAX_WINDOW_SEGMENTS', '3'))


# --------------------------------------------------------------------------- #
# Text normalisation shared by retrieval, coverage and numeric checks
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset("""
a an the this that these those it its it's itself
is was were are be been being am do does did doing done will would should
shall can could may might must
i you he she we they him her them his hers our ours your yours their theirs
and or but so if then than because as while
of to for on in at by with from into onto over under about above below
not no nor
patient doctor physician nurse clinician
right correct okay ok isn't wasn't weren't aren't doesn't didn't don't
there here
q question ask asked asking
have has had
went go going gone come came coming
today yesterday now currently
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

_ONES = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90,
}
_FRACTIONS = {'half': 0.5, 'quarter': 0.25}


def _strip_punct(word: str) -> str:
    return word.strip('.,!?;:"\'')


def normalize_number_words(text: str) -> str:
    """Turn spoken numbers ("two weeks", "one hundred") into digits.

    Whisper sometimes transcribes a number as words and sometimes as digits
    depending on how it was spoken, so comparing digit strings only would miss
    half of the matches. Best-effort: it does not handle every English
    numeral construction, only the ones likely to show up in a dose or a
    duration.
    """
    words = text.split()
    out: List[str] = []
    i, n = 0, len(words)

    while i < n:
        first = _strip_punct(words[i]).lower()

        if first in _ONES or first in _TENS or first in _FRACTIONS:
            value = 0.0
            current = 0.0
            matched = False
            j = i

            while j < n:
                w = _strip_punct(words[j]).lower()
                if w in _ONES:
                    current += _ONES[w]
                    matched = True
                    j += 1
                elif w in _TENS:
                    current += _TENS[w]
                    matched = True
                    j += 1
                elif w == 'hundred':
                    current = (current or 1) * 100
                    matched = True
                    j += 1
                elif w == 'thousand':
                    value += (current or 1) * 1000
                    current = 0
                    matched = True
                    j += 1
                elif w == 'and' and matched:
                    j += 1
                elif w in _FRACTIONS and not matched:
                    value = _FRACTIONS[w]
                    matched = True
                    j += 1
                    break
                else:
                    break

            if matched:
                total = value + current
                text_value = str(int(total)) if total == int(total) else str(total)
                trailing = ''
                for ch in reversed(words[j - 1]):
                    if ch in '.,!?;:':
                        trailing = ch + trailing
                    else:
                        break
                out.append(text_value + trailing)
                i = j
                continue

        out.append(words[i])
        i += 1

    return ' '.join(out)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stem(word: str) -> str:
    """A deliberately crude suffix strip, just enough to line up plurals and
    simple verb forms across the question and the transcript without pulling
    in a real stemmer as a dependency."""
    for suffix in ('ing', 'edly', 'ies', 'ied', 'es', 'ed', 'ly', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> List[str]:
    """Stopword-free, stemmed tokens used for both retrieval and coverage."""
    normalized = normalize_number_words(text)
    return [
        _stem(token) for token in _tokenize(normalized)
        if token not in _STOPWORDS
    ]


def _word_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    # Tolerant of small ASR spelling/stemming mismatches, not of unrelated
    # words: a short common ratio threshold would let too much through.
    return len(a) >= 4 and len(b) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84


def coverage(question_words: List[str], passage_words: List[str]) -> float:
    """Fraction of the question's content words found in the passage."""
    if not question_words:
        return 1.0

    matched = sum(
        1 for qw in question_words
        if any(_word_matches(qw, pw) for pw in passage_words)
    )
    return matched / len(question_words)


# --------------------------------------------------------------------------- #
# Numeric claim verification
# --------------------------------------------------------------------------- #

_UNIT_CANON = {
    'mg': 'mg', 'milligram': 'mg', 'milligrams': 'mg',
    'mcg': 'mcg', 'microgram': 'mcg', 'micrograms': 'mcg',
    'g': 'g', 'gram': 'g', 'grams': 'g',
    'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
    'millilitre': 'ml', 'millilitres': 'ml',
    'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
    'cm': 'cm', 'centimeter': 'cm', 'centimeters': 'cm',
    'centimetre': 'cm', 'centimetres': 'cm',
    'mm': 'mm', 'millimeter': 'mm', 'millimeters': 'mm',
    '%': '%', 'percent': '%',
    'degree': 'degree', 'degrees': 'degree',
    'day': 'day', 'days': 'day',
    'week': 'week', 'weeks': 'week',
    'month': 'month', 'months': 'month',
    'year': 'year', 'years': 'year',
    'hour': 'hour', 'hours': 'hour', 'hr': 'hour', 'hrs': 'hour',
    'minute': 'minute', 'minutes': 'minute', 'min': 'minute', 'mins': 'minute',
    'time': 'time', 'times': 'time',
    'bpm': 'bpm', 'mmhg': 'mmhg',
    'unit': 'unit', 'units': 'unit',
    'dose': 'dose', 'doses': 'dose',
    'tablet': 'tablet', 'tablets': 'tablet',
    'pill': 'pill', 'pills': 'pill',
    'capsule': 'capsule', 'capsules': 'capsule',
}

_UNIT_PATTERN = re.compile(
    r'(?P<num>\d+(?:\.\d+)?)\s*'
    r'(?P<unit>' + '|'.join(sorted(_UNIT_CANON, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


def extract_number_units(text: str) -> List[Tuple[float, str]]:
    """Every (value, canonical unit) pair mentioned in ``text``."""
    normalized = normalize_number_words(text)
    pairs = []
    for match in _UNIT_PATTERN.finditer(normalized):
        unit = _UNIT_CANON[match.group('unit').lower()]
        pairs.append((float(match.group('num')), unit))
    return pairs


_NEGATION_CUES = ('no', 'not', "n't", 'without', 'never', 'none', 'denies', 'denied')


def _has_phrase(normalized_text: str, phrases: List[str]) -> bool:
    """Whether any of ``phrases`` appears, and is not itself negated.

    A bare trigger word like "abnormal" should not count as a match inside
    "no abnormal findings" — that phrase belongs to the opposite side, and is
    matched there as a whole phrase instead. Only checking the four words
    immediately before the match keeps this cheap and avoids the general
    problem of negation scope.
    """
    for phrase in phrases:
        # A word-start boundary, not a plain substring search: "abnormal"
        # must still match "abnormalities", but "normal" must not match
        # inside "abnormal" — "ab-" is not a separator, so anchoring only the
        # left edge lets the first through while still refusing the second.
        pattern = re.compile(r'\b' + re.escape(phrase))

        for match in pattern.finditer(normalized_text):
            preceding_words = normalized_text[max(0, match.start() - 40):match.start()].split()[-4:]
            if not any(cue in preceding_words for cue in _NEGATION_CUES):
                return True

    return False


# Not every hard negative swaps a number — "on an empty stomach" for "after a
# meal" is the README's own example, and lab-finding questions swap "normal"
# for "abnormal" the same way. Each pair below is two families of phrases
# that cannot both be true about the same instruction or finding, so the
# question stating one and the passage stating the other is a contradiction,
# not a paraphrase. Deliberately narrow: anything with everyday, non-medical
# senses (e.g. "left"/"right") is left out because it would misfire on
# unrelated uses of the word.
_OPPOSING_PHRASES: List[Tuple[List[str], List[str]]] = [
    (
        ['empty stomach', 'fasting', 'on an empty stomach'],
        ['after a meal', 'after meals', 'after eating', 'with food', 'with a meal'],
    ),
    (
        ['before meals', 'before eating', 'before food'],
        ['after meals', 'after eating', 'after a meal'],
    ),
    (
        ['normal', 'unremarkable', 'no abnormalities', 'no abnormal findings',
         'within normal limits'],
        ['abnormal', 'elevated', 'irregular', 'abnormalities detected'],
    ),
    (['positive'], ['negative']),
    (['improved', 'improving', 'getting better'], ['worse', 'worsening', 'deteriorating']),
]


def categorical_verdict(question_text: str, passage_text: str) -> Optional[bool]:
    """Like :func:`numeric_verdict`, but for the hard negatives that swap a
    qualitative word instead of a number — see ``_OPPOSING_PHRASES``.
    """
    question_norm = ' ' + normalize_number_words(question_text).lower() + ' '
    passage_norm = ' ' + normalize_number_words(passage_text).lower() + ' '

    confirmed = False
    contradicted = False

    for group_a, group_b in _OPPOSING_PHRASES:
        question_has_a = _has_phrase(question_norm, group_a)
        question_has_b = _has_phrase(question_norm, group_b)
        if question_has_a == question_has_b:
            continue  # the question does not take a side on this pair

        claimed, opposite = (group_a, group_b) if question_has_a else (group_b, group_a)
        passage_has_claimed = _has_phrase(passage_norm, claimed)
        passage_has_opposite = _has_phrase(passage_norm, opposite)

        if passage_has_opposite and not passage_has_claimed:
            contradicted = True
        elif passage_has_claimed:
            confirmed = True

    if contradicted:
        return False
    if confirmed:
        return True
    return None


def numeric_verdict(
    question_pairs: List[Tuple[float, str]],
    passage_text: str,
) -> Optional[bool]:
    """Whether the passage confirms, contradicts, or is silent on the
    question's numeric claims.

    Returns ``True`` if at least one claimed (value, unit) is present in the
    passage and none are contradicted, ``False`` if the passage states a
    different value for a unit the question makes a claim about (the
    "100 mg" vs "200 mg" case), and ``None`` if the question makes no numeric
    claim the passage can confirm or deny either way.
    """
    if not question_pairs:
        return None

    by_unit: Dict[str, List[float]] = {}
    for value, unit in extract_number_units(passage_text):
        by_unit.setdefault(unit, []).append(value)

    confirmed = False
    contradicted = False

    for value, unit in question_pairs:
        candidates = by_unit.get(unit)
        if not candidates:
            continue
        if any(abs(candidate - value) < 1e-6 for candidate in candidates):
            confirmed = True
        else:
            contradicted = True

    if contradicted:
        return False
    if confirmed:
        return True
    return None


# --------------------------------------------------------------------------- #
# Candidate evidence windows
# --------------------------------------------------------------------------- #

def _build_windows(segments: List[Dict]) -> List[Dict]:
    """Every run of 1..MAX_WINDOW_SEGMENTS consecutive segments, as a
    candidate evidence span. Merging is cheap here and lets a two- or
    three-sentence answer win over any single sentence inside it."""
    windows = []
    n = len(segments)

    for size in range(1, min(MAX_WINDOW_SEGMENTS, n) + 1):
        for i in range(0, n - size + 1):
            chunk = segments[i:i + size]
            windows.append({
                'start': chunk[0]['start'],
                'end': chunk[-1]['end'],
                'text': ' '.join(s['text'] for s in chunk),
                'first': i,
                'last': i + size - 1,
            })

    return windows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def answer_questions(
    segments: List[Dict],
    questions: List[str],
) -> List[Tuple[bool, Optional[Span]]]:
    """Answer every question about one transcript.

    ``segments`` is the ASR output: a list of ``{'start', 'end', 'text'}``
    dicts in chronological order. Never raises — a single bad question falls
    back to ``(False, None)`` rather than taking the whole conversation down
    with it.
    """
    if not segments or not questions:
        return [(False, None) for _ in questions]

    windows = _build_windows(segments)

    try:
        vectorizer = TfidfVectorizer(
            tokenizer=content_tokens, token_pattern=None, ngram_range=(1, 1),
        )
        tfidf = vectorizer.fit_transform(
            [w['text'] for w in windows] + list(questions)
        )
    except ValueError:
        # Empty vocabulary — e.g. a transcript that is all stopwords/noise.
        logger.warning('Empty TF-IDF vocabulary; answering "no" to everything.')
        return [(False, None) for _ in questions]

    window_vectors = tfidf[:len(windows)]
    question_vectors = tfidf[len(windows):]
    similarities = cosine_similarity(question_vectors, window_vectors)

    results = []
    for row, question in zip(similarities, questions):
        try:
            results.append(_answer_one(question, row, windows, segments))
        except Exception:
            logger.exception('Falling back to "no" for: %s', question)
            results.append((False, None))

    return results


def _answer_one(
    question: str,
    similarities,
    windows: List[Dict],
    segments: List[Dict],
) -> Tuple[bool, Optional[Span]]:
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return False, None

    best_window = windows[best_idx]

    # A little context on either side of the best window, purely for the
    # detail-verification step below — the evidence span returned is still
    # the tight window itself, not this padded region.
    lo = max(0, best_window['first'] - 1)
    hi = min(len(segments) - 1, best_window['last'] + 1)
    region_text = ' '.join(s['text'] for s in segments[lo:hi + 1])

    question_pairs = extract_number_units(question)
    detail_verdicts = [
        numeric_verdict(question_pairs, region_text),
        categorical_verdict(question, region_text),
    ]

    if False in detail_verdicts:
        return False, None

    if True in detail_verdicts:
        return True, (best_window['start'], best_window['end'])

    question_words = content_tokens(question)
    region_words = content_tokens(region_text)

    if coverage(question_words, region_words) >= COVERAGE_THRESHOLD:
        return True, (best_window['start'], best_window['end'])

    return False, None
