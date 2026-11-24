"""Phase 7: retrieval. Find the clinic notes that answer a caller's question.

The documents in knowledge/ are split into one chunk per section, embedded once at start-up, and searched
with a dot product. The bot calls search() on each caller turn and hands the matching notes to the LLM
along with the question, so answering from the notes costs no extra LLM round trip.

    python rag.py --check              quick self-check
    python rag.py --eval               retrieval accuracy on evals/retrieval_questions.yaml
    python rag.py "do you take cigna"  show what would be retrieved
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "BAAI/bge-small-en-v1.5"  # 133 MB, 384 dimensions, about 6 ms per question on a CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # this model expects it on questions
# Both chosen from `python rag.py --eval`; see that output for the numbers behind them.
# MIN_SCORE is a gate on the best match only: if the best chunk reaches it, the caller is asking about the
# clinic and all TOP_K chunks go to the LLM. An earlier version dropped every chunk below the threshold
# individually, and a voice eval caught it failing: for "anywhere to leave my car" the Parking section
# ranked second at 0.54, just under a 0.55 cut-off, so the bot said it had no information about parking.
# Keyword search (BM25) was tried too. Alone it ranked the right section first for 16 of the first 34
# questions against 30 for embeddings, and mixed into the embedding score it made ranking worse.
#
# MARGIN drops chunks that score far below the best one. Within 0.10 of the best kept the right section
# for all 44 answerable questions while cutting the chunks sent from 5 to 3 on average; 0.07 lost one.
TOP_K = 5
MIN_SCORE = 0.5
MARGIN = 0.10

# Turns that need no clinic facts: booking, names, numbers, yes and no. Written without looking at the test
# set's wording. Facts are skipped when a turn is closer to one of these than to any chunk by ROUTE_MARGIN.
# On 69 labelled turns, 0.25 kept the facts for all 49 questions and sent nothing for 14 of the 20 turns that
# need none, against 6 of 20 without it. The margin is thin: "are you open on saturdays" keeps its facts by
# 0.04, and "please move that appointment to two PM" still gets some. Raise it if questions lose their facts.
NOT_A_QUESTION = [
    "I'd like to make an appointment", "Can you book me in for a check-up on Friday", "I want to schedule a cleaning next week",
    "Could I get an appointment on Monday afternoon", "Please change my appointment to a later time", "I need to move my booking",
    "My name is John Smith", "It's Maria Gonzalez speaking", "My phone number is 212 555 0198", "You can call me back on 646 555 0123",
    "Two one two, five five five", "Yes", "Yes please", "That's right", "No, that's not right", "Sounds good", "Okay great",
    "Thank you very much", "Goodbye", "Hello there", "Hi, good morning", "Sorry, could you say that again", "Hold on one second",
    "I meant the Thursday after that", "Half past ten works for me", "Three in the afternoon please", "The earliest one is fine",
    "Wrong number, sorry", "Can you hear me okay",
]
ROUTE_MARGIN = 0.25

# How the retrieved chunks reach the LLM: inside tags that the system prompt explains. Earlier versions used
# a plain-text header, and the bot told callers what "the clinic notes say" (still 1 reply in 90 after a first
# fix, because the system prompt itself called them "clinic notes").
NOTES_HEADER = "<clinic_facts>"
NOTES_FOOTER = "</clinic_facts>"


def load_chunks(folder=ROOT / "knowledge"):
    """One chunk per '## ' section, with the document title in front so each chunk stands alone."""
    chunks = []
    for path in sorted(folder.glob("*.md")):
        title, section, body = path.stem, None, []
        for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:  # the extra heading flushes the last section
            if line.startswith("## "):
                if section and body:
                    chunks.append({"doc": path.stem, "section": section, "text": f"{title}: {section}. {' '.join(body)}"})
                section, body = line[3:].strip(), []
            elif line.startswith("# "):
                title = line[2:].strip()
            elif line.strip():
                body.append(line.strip())
    return chunks


class Index:
    """The embedded knowledge base.

    ponytail: every chunk is compared with every question in one numpy dot product. That is instant for
    a few thousand chunks; a vector database earns its place only well beyond that.
    """

    def __init__(self, folder=ROOT / "knowledge"):
        self.tokenizer = Tokenizer.from_file(hf_hub_download(MODEL_REPO, "tokenizer.json"))
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # leave the other cores to speech-to-text and the voice
        self.session = ort.InferenceSession(hf_hub_download(MODEL_REPO, "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])
        self.chunks = load_chunks(folder)
        self.vectors = self.embed([c["text"] for c in self.chunks])
        self.not_a_question = self.embed(NOT_A_QUESTION)
        self._verdicts = {}  # piece of a caller's turn -> whether it asks for facts

    def embed(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], np.int64)
        mask = np.array([e.attention_mask for e in encoded], np.int64)
        hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        cls = hidden[:, 0]  # this model's sentence vector is its first token
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)

    def search(self, question, k=TOP_K, min_score=MIN_SCORE, margin=MARGIN, route=True):
        """Up to k best-matching chunks as (score, chunk), best first.

        Empty if even the best chunk is below min_score, or (with `route`) if the turn reads as booking or
        small talk rather than a question. Otherwise every chunk within `margin` of the best.
        """
        as_query, as_sentence = self.embed([QUERY_PREFIX + question, question])  # one batch, so about one embedding's time
        scores = self.vectors @ as_query
        ranked = [(float(scores[i]), self.chunks[i]) for i in np.argsort(-scores)[:k]]
        if ranked[0][0] < min_score:
            return []
        if route and float((self.not_a_question @ as_sentence).max()) >= ranked[0][0] + ROUTE_MARGIN:
            return []
        return [(s, c) for s, c in ranked if s >= ranked[0][0] - margin]

    def asks_for_facts(self, texts):
        """Whether any of these pieces of a caller's turn asks something the clinic facts could answer.

        Each piece is judged on its own, in one batch. A turn as a whole can hide its question: in the evals,
        "...at 10.30 in the morning. My name is Sam Lee. And my number is 555." scored as a question because
        the booking words were diluted, while each of its fragments alone was clearly booking or details.
        """
        texts = [t for t in texts if re.search(r"[A-Za-z]{3}", t)]  # digits alone: a phone number read in pieces
        new = [t for t in dict.fromkeys(texts) if t not in self._verdicts]
        if new:  # each fragment is judged once: a turn re-checks on every new fragment, so 5 fragments took 90 ms
            vectors = self.embed([QUERY_PREFIX + t for t in new] + new)
            best = (vectors[:len(new)] @ self.vectors.T).max(axis=1)
            closest_non_question = (vectors[len(new):] @ self.not_a_question.T).max(axis=1)
            if len(self._verdicts) > 10_000:  # ponytail: crude bound for a long-running server; an LRU if it matters
                self._verdicts.clear()
            self._verdicts.update(zip(new, (bool(x) for x in (best >= MIN_SCORE) & (closest_non_question < best + ROUTE_MARGIN))))
        return any(self._verdicts[t] for t in texts)


def is_notes(message):
    return isinstance(message, dict) and message.get("role") == "developer" and str(message.get("content", "")).startswith(NOTES_HEADER)


def notes_for_turn(index, messages):
    """Attach the notes for the caller's latest turn: (query, chunks found, new message list or None).

    `messages` is the conversation as Pipecat stores it. The caller's turn is every user message since the
    bot last spoke, because speech arrives in fragments and the question is not always the last one: in a
    real eval call "My name is Sam Lee", "And my number is 555.", "014." and "2-2-3-7." were four messages.

    The turn's notes always end up in one block after the whole turn. A version that attached notes after
    each fragment split a phone number across three blocks of notes, and the bot then said the number it
    heard was incomplete. So when another fragment arrives, this turn's earlier block is moved to the end and
    merged with anything new. Only this turn is touched; facts sent in an earlier turn are not sent again.
    Returns None for the message list when nothing needs to change.
    """
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        return "", [], None  # the greeting instruction, a tool result, or nothing to answer
    start = len(messages)
    while start > 0 and isinstance(messages[start - 1], dict) and messages[start - 1].get("role") in ("user", "developer"):
        start -= 1  # back to the bot's last reply or tool call: the caller's turn starts after it
    turn = messages[start:]
    fragments = [m["content"].strip() for m in turn if m.get("role") == "user" and isinstance(m.get("content"), str)]
    query = " ".join(fragments).strip()
    # Judge each fragment, and each adjacent pair in case a pause split a question in two; if any of them asks
    # for facts, search with the whole turn, without the booking check, which has already been made.
    pieces = fragments + [f"{a} {b}" for a, b in zip(fragments, fragments[1:])]

    def facts(block):
        return [line[2:] for line in block["content"].splitlines() if line.startswith("- ")]

    earlier = {fact for m in messages[:start] if is_notes(m) for fact in facts(m)}
    this_turn = [fact for m in turn if is_notes(m) for fact in facts(m)]
    found = index.search(query, route=False) if index.asks_for_facts(pieces) else []
    merged = list(dict.fromkeys(this_turn + [c["text"] for _, c in found if c["text"] not in earlier]))
    if not merged:
        return query, found, None
    block = {"role": "developer", "content": "\n".join([NOTES_HEADER, *(f"- {fact}" for fact in merged), NOTES_FOOTER])}
    if turn[-1] == block:
        return query, found, None  # already in place
    return query, found, messages[:start] + [m for m in turn if not is_notes(m)] + [block]


def evaluate(index):
    import yaml

    data = yaml.safe_load((ROOT / "evals" / "retrieval_questions.yaml").read_text(encoding="utf-8"))
    answerable, nothing = data["answerable"], data["nothing_expected"]

    # Ranking quality, ignoring the gate and the margin. For each question: the best score overall, the best
    # score of a right section within the top k (0 if none made it), and whether a right section came first.
    rows, top1, missed = [], 0, []
    for item in answerable:
        ranked = index.search(item["q"], min_score=-1, margin=1, route=False)
        right = [score for score, c in ranked if c["section"] in item["sections"]]
        rows.append((ranked[0][0], max(right, default=0.0)))
        top1 += ranked[0][1]["section"] in item["sections"]
        if not right:
            missed.append(item["q"])
    best_when_nothing = [index.search(item["q"], min_score=-1, route=False)[0][0] for item in nothing]

    n, in_top_k = len(answerable), sum(right > 0 for _, right in rows)
    print(f"{len(index.chunks)} chunks from {len({c['doc'] for c in index.chunks})} documents, {n} answerable questions, {len(nothing)} where nothing should match")
    print(f"right section ranked first: {top1}/{n} ({top1 / n:.0%})   in the top {TOP_K}: {in_top_k}/{n} ({in_top_k / n:.0%})")
    for q in missed:
        print(f"  not in the top {TOP_K}: {q!r}")
    print(f"best score for an answerable question: lowest {min(best for best, _ in rows):.2f}")
    print(f"best score when nothing should match: {sorted(round(s, 2) for s in best_when_nothing)}")

    # Two ways to use a threshold. "gate": the best chunk must reach it, then all top-k go to the LLM.
    # "per chunk": every chunk must reach it on its own. The right section has to arrive either way.
    print("threshold   right section reaches the LLM       nothing sent when nothing should match")
    print("            gate (used)    per chunk")
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60):
        gate = sum(best >= threshold and right > 0 for best, right in rows)
        per_chunk = sum(right >= threshold for _, right in rows)
        quiet = sum(s < threshold for s in best_when_nothing)
        mark = "  <- MIN_SCORE" if abs(threshold - MIN_SCORE) < 1e-9 else ""
        print(f"   {threshold:.2f}        {gate}/{n}          {per_chunk}/{n}              {quiet}/{len(nothing)}{mark}")

    # The settings actually used, end to end: gate, margin, routing and all.
    sent = [index.search(item["q"]) for item in answerable]
    kept = sum(any(c["section"] in item["sections"] for _, c in found) for item, found in zip(answerable, sent))
    still = [item["q"] for item in nothing if index.search(item["q"])]
    print(f"as configured (gate {MIN_SCORE}, top {TOP_K}, margin {MARGIN}, route margin {ROUTE_MARGIN}): right section "
          f"sent for {kept}/{n}, {np.mean([len(f) for f in sent]):.1f} chunks on average; nothing sent for "
          f"{len(nothing) - len(still)}/{len(nothing)} turns that need none")
    for q in still:
        print(f"  facts still sent for: {q!r}")

    # Speech arrives in fragments, so the question is not always the last thing said in a turn.
    shapes = {"question last": lambda q: ["Hi there.", q], "question first": lambda q: [q, "Thanks."],
              "question in the middle": lambda q: ["Hi, quick question.", q, "Just wondering."]}
    greeting = [{"role": "developer", "content": "Greet the caller."}, {"role": "assistant", "content": "Hello, how can I help?"}]
    for name, build in shapes.items():
        hits = 0
        for item in answerable:
            _, _, after = notes_for_turn(index, greeting + [{"role": "user", "content": p} for p in build(item["q"])])
            sent = after[-1]["content"] if after else ""
            hits += any(c["text"] in sent for c in index.chunks if c["section"] in item["sections"])
        print(f"caller turn in fragments, {name}: right section sent for {hits}/{n}")


def check(index):
    chunks = load_chunks()
    assert len(chunks) >= 25 and all(c["text"] and c["section"] for c in chunks), len(chunks)
    assert len({(c["doc"], c["section"]) for c in chunks}) == len(chunks), "duplicate section headings"
    assert index.vectors.shape == (len(chunks), 384), index.vectors.shape

    found = {chunk["section"]: chunk["text"] for _, chunk in index.search("how much is a teeth cleaning")}
    assert "95 dollars" in found.get("Routine cleaning", ""), found  # in the results, though not always first
    assert index.search("yes that's correct") == [], "small talk should retrieve nothing"
    assert index.search("I'd like to book a cleaning for next Wednesday at half past ten") == [], "a booking needs no facts"
    assert index.search("I'd like to book a cleaning, how much does that cost"), "a booking with a question still gets facts"

    # notes_for_turn, on conversations shaped the way Pipecat stores them.
    start = [{"role": "developer", "content": "Greet the caller."}, {"role": "assistant", "content": "Hello, how can I help?"}]
    say = lambda *parts: [{"role": "user", "content": p} for p in parts]
    roles = lambda msgs: "".join("N" if is_notes(m) else m["role"][0] for m in msgs)  # d, a, u, N for notes

    after = notes_for_turn(index, start + say("how much is a teeth cleaning"))[2]
    assert roles(after) == "dauN" and "95 dollars" in after[-1]["content"], roles(after)
    after = notes_for_turn(index, start + say("How much is a teeth cleaning?", "Thanks."))[2]
    assert after and "95 dollars" in after[-1]["content"], "a question followed by another fragment must still be found"

    # A turn that keeps growing: the caller's fragments must stay together, with one notes block at the end.
    turn = notes_for_turn(index, start + say("How much is a teeth cleaning?"))[2]
    for fragment in ("My number is 555.", "014", "2-2-3-7."):
        turn = notes_for_turn(index, turn + say(fragment))[2] or turn + say(fragment)
    assert roles(turn) == "dauuuuN", f"fragments split by notes: {roles(turn)}"
    assert "95 dollars" in turn[-1]["content"], "facts found earlier in the turn must be kept"
    assert notes_for_turn(index, turn)[2] is None, "nothing to change when the notes are already in place"

    # Turns copied from eval call logs: booking, name and number in fragments get no facts; a question split
    # across two fragments, or asked after booking words, still does.
    booking = say("Hi, I'd like to book a cleaning for next Wednesday at 10.30 in the morning.", "My name is Sam Lee",
                  "And my number is 555.", "014.", "2-2-3-7.")
    assert notes_for_turn(index, start + booking)[2] is None, "a booking turn in fragments must get no facts"
    split = notes_for_turn(index, start + say("How much does a", "routine cleaning cost?"))[2]
    assert split and "95 dollars" in split[-1]["content"], "a question split by a pause must still get facts"
    mixed = notes_for_turn(index, start + say("I'd like to book a cleaning next week.", "My name is Sam.", "How much is it?"))[2]
    assert mixed and "95 dollars" in mixed[-1]["content"], "a question after booking words must still get facts"

    # Facts sent in an earlier turn are not sent again; a tool call ends the caller's turn.
    again = notes_for_turn(index, turn + [{"role": "assistant", "content": "It is 95 dollars."}] + say("How much is a cleaning again?"))[2]
    assert again is None or "95 dollars" not in again[-1]["content"], "a fact from an earlier turn was sent again"
    assert notes_for_turn(index, start + say("2-2-3-7."))[2] is None, "digits alone must not trigger retrieval"
    assert notes_for_turn(index, start + say("yes that's correct"))[2] is None, "small talk must not trigger retrieval"
    assert notes_for_turn(index, start)[2] is None, "nothing to do when the bot spoke last"
    assert notes_for_turn(index, start + say("how much is a cleaning") + [{"role": "tool", "content": "{}"}])[2] is None
    assert notes_for_turn(index, [])[2] is None
    print(f"retrieval ok: {len(chunks)} chunks, a cleaning question retrieves {sorted(found)}, turn handling ok")


if __name__ == "__main__":
    index = Index()
    if "--check" in sys.argv:
        check(index)
    elif "--eval" in sys.argv:
        evaluate(index)
    else:
        for score, chunk in index.search(" ".join(sys.argv[1:]), min_score=-1):
            print(f"{score:.2f}  [{chunk['doc']} / {chunk['section']}]  {chunk['text'][:110]}")
