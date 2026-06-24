from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import numpy as np
import re
import os

app = Flask(__name__)

# ============================================================
#  LOAD LSTM MODEL + IMDB WORD INDEX
# ============================================================

VOCAB_SIZE = 5000
MAXLEN     = 200

print("Loading IMDB word index...")
word_index = imdb.get_word_index()

print("Loading LSTM model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.h5")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


# ============================================================
#  NEGATION WORDS
# ============================================================

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "cant", "can't", "won't",
    "wont", "doesn't", "doesnt", "didn't", "didnt",
    "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
    "hardly", "scarcely", "barely", "without", "lack", "lacking"
}

# ============================================================
#  SENTIMENT WORD LISTS
# ============================================================

POSITIVE_WORDS = {
    "excellent", "amazing", "outstanding", "superb", "fantastic",
    "wonderful", "brilliant", "perfect", "incredible", "extraordinary",
    "magnificent", "phenomenal", "spectacular", "marvelous", "terrific",
    "masterpiece", "flawless", "exceptional", "remarkable", "breathtaking",
    "good", "great", "nice", "interesting", "enjoyable", "entertaining",
    "impressive", "beautiful", "lovely", "exciting", "fun", "awesome",
    "cool", "solid", "liked", "love", "loved", "enjoy", "enjoyed",
    "fascinating", "engaging", "captivating", "touching", "moving",
    "charming", "delightful", "pleasing", "satisfying", "rewarding",
    "recommend", "recommended", "worth", "watchable", "decent",
    "happy", "glad", "pleased", "thrilled", "inspired",
}

NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "dreadful", "atrocious",
    "appalling", "disgusting", "pathetic", "abysmal", "disastrous",
    "unbearable", "unwatchable", "insufferable",
    "bad", "poor", "boring", "worst", "waste", "hate", "hated",
    "disappointing", "disappointed", "mediocre", "weak", "slow",
    "dull", "stupid", "annoying", "ugly", "failed", "failure",
    "garbage", "trash", "rubbish", "useless", "pointless",
    "predictable", "cliche", "overrated", "forgettable", "bland",
    "confusing", "messy", "incoherent", "offensive", "ridiculous",
    "laughable", "painful", "tedious", "sloppy",
}

# ============================================================
#  PHRASE LISTS  (score, phrase)
#  KEY:
#    score  0  = neutral signal  (e.g. "not bad" — neither good nor bad)
#    score +1/+2/+3 = mild/moderate/strong positive
#    score -1/-2/-3 = mild/moderate/strong negative
# ============================================================

PHRASES = [
    # --- Neutral phrases (cancel out to neutral) ---
    ( 0, "not bad"),
    ( 0, "not too bad"),
    ( 0, "not so bad"),
    ( 0, "not great not terrible"),
    ( 0, "nothing special"),

    # --- Positive phrases ---
    ( 2, "pretty good"),
    ( 2, "quite good"),
    ( 2, "very good"),
    ( 2, "really good"),
    ( 2, "so good"),
    ( 3, "too good"),
    ( 3, "damn good"),
    ( 2, "very nice"),
    ( 2, "really nice"),
    ( 2, "very interesting"),
    ( 3, "highly recommend"),
    ( 3, "must watch"),
    ( 3, "must see"),
    ( 2, "worth watching"),
    ( 2, "worth seeing"),
    ( 2, "surprisingly good"),
    ( 2, "pleasantly surprised"),
    ( 2, "better than expected"),
    ( 2, "could not stop watching"),
    ( 2, "couldn't stop watching"),
    ( 2, "can't stop watching"),

    # --- Negative phrases ---
    (-2, "not good"),
    (-2, "not great"),
    (-2, "not worth"),
    (-2, "not enjoyable"),
    (-3, "waste of time"),
    (-3, "waste of money"),
    (-3, "not recommended"),
    (-3, "do not watch"),
    (-3, "don't watch"),
    (-3, "complete waste"),
    (-3, "total waste"),
    (-3, "utter waste"),
    (-2, "not impressive"),
    (-2, "not interesting"),
    (-2, "not entertaining"),
    (-2, "did not enjoy"),
    (-2, "did not like"),
    (-2, "could not enjoy"),
    (-2, "couldn't enjoy"),
    (-2, "doesn't make sense"),
    (-2, "made no sense"),
]

# Sort longest phrase first so multi-word matches take priority
PHRASES.sort(key=lambda x: -len(x[1]))


# ============================================================
#  NEGATION-AWARE KEYWORD SCORER
# ============================================================

def keyword_score(text):
    """
    Returns (score, has_negation):
      score        = signed int — raw strength, NOT divided by word_count
      has_negation = True if any negation word found

    Phrase matching runs first (longest→shortest).
    Negation window = 4 words after a negation trigger.
    """
    text_lower   = text.lower()
    score        = 0
    has_negation = False
    used_spans   = []

    # ---- Phase 1: phrase matching ----
    for val, phrase in PHRASES:
        for m in re.finditer(re.escape(phrase), text_lower):
            # Don't double-count overlapping spans
            span = (m.start(), m.end())
            if not any(ps <= m.start() < pe for ps, pe in used_spans):
                score += val
                used_spans.append(span)

    # ---- Phase 2: single-word scan with negation window ----
    word_tokens = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"[a-z']+", text_lower)
    ]

    negation_active    = False
    negation_countdown = 0

    for word, wstart, wend in word_tokens:

        if word in NEGATION_WORDS:
            negation_active    = True
            negation_countdown = 4
            has_negation       = True
            continue

        if negation_active:
            negation_countdown -= 1
            if negation_countdown <= 0:
                negation_active = False

        # Skip chars already covered by a phrase
        if any(ps <= wstart < pe for ps, pe in used_spans):
            continue

        if word in POSITIVE_WORDS:
            contribution = +1
        elif word in NEGATIVE_WORDS:
            contribution = -1
        else:
            continue

        if negation_active:
            contribution = -contribution

        score += contribution

    return score, has_negation


# ============================================================
#  TOKENIZER
# ============================================================

def tokenize(text):
    words  = re.findall(r"[a-z']+", text.lower())
    tokens = []
    for w in words:
        idx = word_index.get(w)
        if idx is None or (idx + 3) >= VOCAB_SIZE:
            tokens.append(2)
        else:
            tokens.append(idx + 3)
    return pad_sequences([tokens], maxlen=MAXLEN)


# ============================================================
#  HYBRID PREDICTOR
# ============================================================

def predict_sentiment(text):
    """
    Short reviews  (≤ 8 words)  → keyword score dominates
    Long  reviews  (> 8 words)  → LSTM dominates, keyword nudges

    Blending (short path):
        kw_signal = clamp(kw * 0.35,  -1, +1)
        blended   = 0.5 + kw_signal          (0.0 – 1.0)
        final     = blended*0.85 + lstm*0.15

    Thresholds:
        > 0.58 → Positive
        < 0.42 → Negative
        else   → Neutral
    """
    words      = re.findall(r"[a-z']+", text.lower())
    word_count = len(words)

    padded     = tokenize(text)
    lstm_score = float(model.predict(padded, verbose=0)[0][0])

    kw, has_negation = keyword_score(text)

    recognised = sum(
        1 for w in words
        if word_index.get(w) is not None and (word_index[w] + 3) < VOCAB_SIZE
    )
    recognition_ratio = recognised / max(word_count, 1)

    # ---- Blend ----
    if word_count <= 8 or recognition_ratio < 0.4:
        kw_signal = max(-1.0, min(1.0, float(kw) * 0.35))
        blended   = max(0.0, min(1.0, 0.5 + kw_signal))
        final     = blended * 0.85 + lstm_score * 0.15
    else:
        kw_signal = max(-0.15, min(0.15, kw * 0.03))
        final     = lstm_score + kw_signal

    final = max(0.0, min(1.0, final))

    # ---- Label ----
    if final > 0.58:
        result     = "Positive"
        confidence = round(final * 100, 1)
    elif final < 0.42:
        result     = "Negative"
        confidence = round((1 - final) * 100, 1)
    else:
        result     = "Neutral"
        confidence = round(50.0 + (0.16 - abs(final - 0.5)) * 200, 1)
        confidence = max(50.0, min(99.0, confidence))

    return result, confidence


# ============================================================
#  HISTORY
# ============================================================

history = []


# ============================================================
#  ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    result     = None
    confidence = None

    if request.method == 'POST':
        text = request.form['review'].strip()
        if text:
            result, confidence = predict_sentiment(text)
            history.append({
                "text"      : text,
                "result"    : result,
                "confidence": confidence,
                "time"      : datetime.now().strftime("%H:%M  %d-%m-%Y")
            })

    return render_template("analyzer.html", result=result, confidence=confidence)


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    text = data.get("review", "").strip()
    if not text:
        return jsonify({"error": "No review text provided"}), 400
    result, confidence = predict_sentiment(text)
    return jsonify({"sentiment": result, "confidence": confidence})


@app.route('/history')
def history_page():
    return render_template("history.html", history=history)


@app.route('/analytics')
def analytics():
    pos = sum(1 for h in history if h['result'] == "Positive")
    neg = sum(1 for h in history if h['result'] == "Negative")
    neu = sum(1 for h in history if h['result'] == "Neutral")
    return render_template("analytics.html", pos=pos, neg=neg, neu=neu)


if __name__ == "__main__":
    app.run(debug=True)