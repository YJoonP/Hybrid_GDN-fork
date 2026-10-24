"""Long-context task generators with gold-span annotation (needed for oracle relevance).

Each sample is a dict:
  prompt      : full chat-formatted prompt string (ends at the assistant answer slot)
  answers     : list[str] of acceptable answers
  spans       : dict name -> list[(char_start, char_end)] in `prompt`
                q_before (question tokens placed before the context, if any),
                context, gold, q_after
  meta        : misc
"""
import json
import random
import string

INSTR = {
    "niah": "Some special magic numbers are hidden in the following text. Memorize them. "
            "Answer with only the number, no other words.",
    "hotpot": "Answer the question based on the given passages. "
              "Only give me the answer and do not output any other words.",
    "kv": "Extract the value corresponding to the specified key in the JSON object below. "
          "Answer with only the value, no other words.",
}

ADJ = ("red blue green quiet loud brave calm eager fancy gentle happy jolly kind lively proud silly witty "
       "zealous ancient bitter clever dusty empty fierce giant hollow icy juicy little mighty narrow odd plain "
       "rapid shiny tiny vast wooden young crisp damp faint grumpy hasty itchy lazy muddy noisy polite rusty").split()
NOUN = ("apple bridge castle dragon engine forest garden harbor island jacket kettle lantern meadow needle orchard "
        "pencil quarry river saddle tunnel umbrella valley wagon yacht zebra anchor barrel candle desert falcon "
        "glacier hammer igloo jungle kitten ladder magnet nectar oyster parrot quilt rocket spider tower violin walrus").split()


class Builder:
    """Accumulates prompt text while recording char spans."""

    def __init__(self):
        self.parts, self.n, self.spans = [], 0, {}

    def add(self, text, span=None):
        if span is not None:
            self.spans.setdefault(span, []).append((self.n, self.n + len(text)))
        self.parts.append(text)
        self.n += len(text)

    def mark_start(self):
        return self.n

    def mark(self, name, start):
        self.spans.setdefault(name, []).append((start, self.n))

    def text(self):
        return "".join(self.parts)


def wrap_chat(build_user, assistant_prefix=""):
    """Qwen3.5 non-thinking chat format; build_user(builder) fills the user turn."""
    b = Builder()
    b.add("<|im_start|>user\n")
    build_user(b)
    b.add("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    b.add(assistant_prefix)
    return b


def layout(b, fmt, instr, question, put_context):
    """fmt in {after, before, both}. put_context(b) writes the context."""
    if fmt in ("before", "both"):
        b.add("Question: ")
        b.add(question, span="q_before")
        b.add("\n\n")
    b.add(instr + "\n\n")
    s = b.mark_start()
    put_context(b)
    b.mark("context", s)
    if fmt in ("after", "both"):
        b.add("\n\nQuestion: ")
        b.add(question, span="q_after")
        b.add("\nAnswer:")
    else:
        b.add("\n\nAnswer the question given at the beginning.\nAnswer:")


def load_paragraphs(path):
    with open(path) as f:
        return json.load(f)


def fill_paragraphs(tok, pool, budget, rng, exclude=()):
    """Sample paragraphs until ~budget tokens."""
    out, n = [], 0
    while n < budget:
        p = pool[rng.randrange(len(pool))]
        if p["title"] in exclude:
            continue
        out.append(p)
        n += p["ntok"]
    return out


def make_niah(tok, pool, ctx_tokens, n_needles, fmt, rng):
    keys = set()
    while len(keys) < n_needles:
        keys.add(f"{rng.choice(ADJ)}-{rng.choice(NOUN)}")
    keys = list(keys)
    vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
    target = rng.randrange(n_needles)
    hay = fill_paragraphs(tok, pool, ctx_tokens - 20 * n_needles, rng)
    # needle insertion points (paragraph boundaries), spread uniformly at random
    slots = sorted(rng.sample(range(len(hay) + 1), n_needles))
    order = list(range(n_needles))
    rng.shuffle(order)
    question = f"What is the special magic number for {keys[target]} mentioned in the provided text?"

    def put(b):
        k = 0
        for i in range(len(hay) + 1):
            while k < n_needles and slots[k] == i:
                j = order[k]
                b.add(f"The special magic number for {keys[j]} is: {vals[j]}.",
                      span="gold" if j == target else "distractor")
                b.add("\n\n")
                k += 1
            if i < len(hay):
                b.add(hay[i]["text"] + "\n\n")

    b = wrap_chat(lambda b: layout(b, fmt, INSTR["niah"], question, put))
    depth = slots[order.index(target)] / max(1, len(hay))
    return dict(prompt=b.text(), answers=[vals[target]], spans=b.spans,
                meta=dict(task="niah", n_needles=n_needles, depth=depth))


def make_hotpot(tok, pool, ex, ctx_tokens, fmt, rng):
    gold_titles = set(ex["supporting_facts"]["title"])
    paras = []
    for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
        text = f"Title: {title}\n" + "".join(sents)
        paras.append(dict(title=title, text=text, gold=title in gold_titles))
    used = sum(len(tok(p["text"]).input_ids) for p in paras)
    extra = fill_paragraphs(tok, pool, ctx_tokens - used, rng, exclude={p["title"] for p in paras})
    paras += [dict(title=p["title"], text=p["text"], gold=False) for p in extra]
    rng.shuffle(paras)

    def put(b):
        for i, p in enumerate(paras):
            b.add(f"Passage {i + 1}:\n")
            b.add(p["text"], span="gold" if p["gold"] else None)
            b.add("\n\n")

    b = wrap_chat(lambda b: layout(b, fmt, INSTR["hotpot"], ex["question"], put))
    gpos = [i / len(paras) for i, p in enumerate(paras) if p["gold"]]
    return dict(prompt=b.text(), answers=[ex["answer"]], spans=b.spans,
                meta=dict(task="hotpot", id=ex["id"], type=ex["type"], depth=gpos))


def _uuid(rng):
    h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def make_kv(tok, ctx_tokens, fmt, rng, tok_per_pair=None):
    tok_per_pair = tok_per_pair or len(tok(f'"{_uuid(rng)}": "{_uuid(rng)}", ').input_ids)
    n = max(2, ctx_tokens // tok_per_pair)
    pairs = [(_uuid(rng), _uuid(rng)) for _ in range(n)]
    target = rng.randrange(n)
    question = f'What is the value of key "{pairs[target][0]}"?'

    def put(b):
        b.add("JSON data:\n{")
        for i, (k, v) in enumerate(pairs):
            b.add(f'"{k}": "{v}"', span="gold" if i == target else None)
            b.add(", " if i < n - 1 else "}")

    b = wrap_chat(lambda b: layout(b, fmt, INSTR["kv"], question, put))
    return dict(prompt=b.text(), answers=[pairs[target][1]], spans=b.spans,
                meta=dict(task="kv", n_pairs=n, depth=target / n))
