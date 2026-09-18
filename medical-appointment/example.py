"""Transcribe the conversation locally, then answer every question against it.

Two stages, run once per request and shared across all ten questions:

1. ``transcribe`` — local ASR (faster-whisper) turns the audio into timestamped
   segments. This is the expensive half and the only one that scales with
   audio length.
2. ``answer_questions`` (in ``answering.py``) — decides each question against
   the transcript and, for every "yes", which segments it was read from. See
   that module's docstring for why it is rule-based rather than another model
   call: the dataset's hard negatives are near-misses on a checkable detail
   (a dose, a duration, a word), not on topic, so verifying the transcript's
   own words gets further than judging semantic similarity would.

Nothing here calls a cloud API. Both the ASR model and the answering logic run
on this machine, as the competition rules require.
"""

import logging
import os
import tempfile
import time
from typing import Dict, List

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio
from answering import answer_questions

logger = logging.getLogger(__name__)

# CPU-safe by default; override for the hardware you actually deploy on.
# A GPU box can go much bigger (e.g. WHISPER_MODEL=large-v3,
# WHISPER_DEVICE=cuda, WHISPER_COMPUTE_TYPE=float16) and get both better
# accuracy and tighter evidence out of the same 60-second budget.
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL', 'small.en')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'auto')
WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
WHISPER_BEAM_SIZE = int(os.environ.get('WHISPER_BEAM_SIZE', '1'))

_model = None


def _get_model():
    """Load the ASR model once and keep it. Called eagerly at import time
    (see the bottom of this file) so the first real request is not the one
    that pays for it — there is no warm-up grace period in the timing rules.
    """
    global _model

    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            'Loading Whisper model %r (device=%s, compute_type=%s)',
            WHISPER_MODEL_NAME, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    return _model


def transcribe(audio_bytes: bytes) -> List[Dict]:
    """The conversation as timestamped segments: ``[{'start', 'end', 'text'}, ...]``.

    Segment-level timing (rather than joining everything into one string) is
    what lets a "yes" answer carry real evidence — see the README's "Keep the
    timings" section.
    """
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
        f.write(audio_bytes)
        f.flush()

        segments, _info = model.transcribe(
            f.name,
            language='en',
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
        )

        return [
            {'start': segment.start, 'end': segment.end, 'text': segment.text.strip()}
            for segment in segments
            if segment.text and segment.text.strip()
        ]


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    The whole conversation and all of its questions arrive together, so the
    expensive half — transcription — is paid once here and shared by every
    answer below.
    """
    started = time.time()
    audio_bytes = decode_audio(request.audio_base64)

    duration = audio_duration_seconds(audio_bytes)
    logger.info(
        '%s (%.1f s, %.1f MB): %d questions',
        request.audio_filename,
        duration if duration is not None else float('nan'),
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    # Never let either stage raise. An exception means no response, and no
    # response means every question about this conversation is scored wrong —
    # ten marks, not one. Falling back to "no evidence found" for every
    # question is worth half a mark on average; an error is worth nothing.
    try:
        segments = transcribe(audio_bytes)
    except Exception:
        logger.exception('Transcription failed for %s; answering blind.',
                          request.audio_filename)
        segments = []

    try:
        results = answer_questions(segments, request.questions)
    except Exception:
        logger.exception('Answering failed for %s; guessing.',
                          request.audio_filename)
        results = [(False, None) for _ in request.questions]

    answers, evidence_start, evidence_end = [], [], []
    for answer, span in results:
        answers.append(bool(answer))
        evidence_start.append(float(span[0]) if span is not None else None)
        evidence_end.append(float(span[1]) if span is not None else None)

    logger.info(
        '%s: answered %d questions in %.1f s',
        request.audio_filename, len(request.questions), time.time() - started,
    )

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )


# Loaded once, at import time, so the first request does not pay for it — see
# the README's note that there is no separate warm-up period.
try:
    _get_model()
except Exception:
    logger.exception(
        'Could not preload the Whisper model at import time; the first '
        'request will try again and may be slow.'
    )
