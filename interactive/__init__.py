"""
interactive/ - ask SENSE questions about your surroundings ("Path B").

    "What colour is the bottle?"     "How many chairs are in front of me?"
    "Is there anything on my right?" "What was in front of me before?"
    "What time is it?"               "Where is it?"  (follow-up)

A separate layer on top of SENSE. It only READS what the main loop already
computed (last YOLO frame, memory database, door model, clock) and answers
through SENSE's own speak(). It adds no thread, opens no camera, runs no
detector and needs no internet. Questions it does not own fall through to
the existing SENSE code unchanged; if this package fails to import, SENSE
runs exactly as before.

Free-form questions the rules cannot handle can go to a local AI model
(Ollama, offline) - see local_llm.py and docs/INTERACTIVE_QA.md.
"""
from . import config
from .local_llm import OllamaLLM
from .qa import InteractiveQA, Reply


def create(hooks, log=print):
    llm = OllamaLLM(log=log) if config.LLM_ENABLED else None
    if llm is None:
        log("💬 Local AI model switched off (SENSE_LLM=off). Rule-based answers only.")
    qa = InteractiveQA(hooks, llm=llm, log=log)
    log("💬 Interactive questions on: colour, how many, left/right, describe, earlier, date and time.")
    return qa


__all__ = ["create", "InteractiveQA", "Reply", "OllamaLLM"]
