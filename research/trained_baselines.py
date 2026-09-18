"""
research/trained_baselines.py

Trained non-LLM severity classifiers. Design fixed in advance in
research/AEI/analysis_plan_additions.md (section A).

Every classifier here decides severity only. Sensor routing and spike value come
from the same keyword routing and band-centre values as the regex baseline
(`baselines._keyword_spike_with_severity`), so all non-LLM arms differ only in
how severity is decided.

Training regimes
    R1  prompt knowledge: synthetic sentences built only from vocabulary the
        production prompt gives the LLM (the severity term lists in baselines.py
        and the fault phrases of the prompt's FAULT → SENSOR table). No test
        prompt is used, so R1 models can be applied to every test set.
    R2  in-domain: R1 plus labelled prompts from both paraphrase banks, excluding
        the test prompt's cluster (leave-one-anchor-out on the anchored bank,
        leave-one-prompt-out on the design bank). In-domain examples are weighted
        so both sources carry equal total weight. Arm names get `_indomain`.

R1 labels follow the production prompt's rules, the precedence the regex uses:
HIGH if the sentence contains a HIGH term or the fault phrase's table line is
annotated "(HIGH severity)", else LOW if it contains a LOW term, else MEDIUM.

Models (hyperparameters fixed before any result, no tuning on test data)
    tfidf_logreg      TF-IDF word 1-2-grams + char 2-5-grams, logistic regression
    minilm_logreg     all-MiniLM-L6-v2 embeddings, logistic regression
    bge_m3_logreg     BAAI/bge-m3 dense embeddings (multilingual), logistic regression
    minilm_finetuned  all-MiniLM-L6-v2 fine-tuned end to end with a linear head

Fitted R1 models and training-text embeddings are cached in
research/trained_baselines_artifacts/ (gitignored, rebuilt deterministically).

Usage:
    python -m research.trained_baselines                   # fit R1 models, write the training-set summary
    python -m research.trained_baselines --bank anchored   # also R2 cross-validated predictions for a bank

Output:
    research/results/trained_baselines/training_set_summary.md
    research/results/trained_baselines/indomain_predictions_{design,anchored}.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARMS = ["tfidf_logreg", "minilm_logreg", "bge_m3_logreg", "minilm_finetuned"]
REFERENCE_ARM = "bge_m3_logreg"
IN_DOMAIN_SUFFIX = "_indomain"
EXTERNAL_SUFFIX = "_external"   # R3: R1 sentences + every labelled bank prompt, applied to another site's text
LABELS = ["LOW", "MEDIUM", "HIGH"]

TEMPLATE_SEED = 20260917
TEMPLATES = [
    "{term} {fault} on Machine {m}",
    "{fault} {term} on Machine {m}",
    "Machine {m}: {term} {fault}",
    "{term} {fault} detected on Machine {m}",
]
BARE_TEMPLATES = ["{fault} on Machine {m}", "{fault} detected on Machine {m}"]

ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge_m3": "BAAI/bge-m3"}
LOGREG_C = 1.0
FT_EPOCHS, FT_LR_ENCODER, FT_LR_HEAD, FT_BATCH, FT_SEED, FT_MAX_LEN = 4, 3e-5, 1e-3, 32, 0, 64
FT_DEVICE = "cpu"   # training device; R1 was trained on CPU, R2 folds on CUDA (--device cuda) for time

ARTIFACT_DIR = PROJECT_ROOT / "research" / "trained_baselines_artifacts"
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "trained_baselines"


# ─────────────────────────────────────────────────────────────────────────────
# Training data
# ─────────────────────────────────────────────────────────────────────────────

def fault_phrases_from_prompt() -> list[tuple[str, str | None]]:
    """(phrase, severity annotation or None) for every left-hand phrase in the FAULT → SENSOR table."""
    from agents.prompts import DIAGNOSTIC_SYSTEM_PROMPT

    block = DIAGNOSTIC_SYSTEM_PROMPT.split("== FAULT → SENSOR MAPPING")[1].split("==")[1]
    phrases: dict[str, str | None] = {}
    for line in block.splitlines():
        if "→" not in line:
            continue
        left = line.split("→")[0].strip()
        if left.startswith("If "):
            continue
        annotation = re.search(r"\((HIGH|MEDIUM) severity\)", left)
        for phrase in re.sub(r"\(.*?\)", "", left).split("/"):
            phrase = phrase.strip()
            if phrase and phrase not in phrases:
                phrases[phrase] = annotation.group(1) if annotation else None
    return list(phrases.items())


def prompt_rule_label(text: str, annotation: str | None) -> str:
    from research.baselines import _HIGH_PATTERN, _LOW_PATTERN

    if annotation == "HIGH" or _HIGH_PATTERN.search(text):
        return "HIGH"
    if _LOW_PATTERN.search(text):
        return "LOW"
    return "MEDIUM"


def template_training_set() -> pd.DataFrame:
    """Regime R1 training sentences (deterministic)."""
    from research.baselines import _HIGH_TERMS, _LOW_TERMS, _MEDIUM_TERMS

    rng = np.random.default_rng(TEMPLATE_SEED)
    faults = fault_phrases_from_prompt()
    rows = []
    for terms in (_HIGH_TERMS, _LOW_TERMS, _MEDIUM_TERMS):
        for term in terms:
            for fault, annotation in faults:
                template = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
                text = template.format(term=term, fault=fault, m=int(rng.integers(1, 6)))
                rows.append({"text": text, "label": prompt_rule_label(text, annotation)})
    for fault, annotation in faults:
        for template in BARE_TEMPLATES:
            text = template.format(fault=fault, m=int(rng.integers(1, 6)))
            rows.append({"text": text, "label": prompt_rule_label(text, annotation)})
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


def labelled_bank_prompts() -> pd.DataFrame:
    """Every prompt of the design and anchored banks with its inherited or design label."""
    from research.paraphrase.anchored_bank import ANCHORED_PROMPTS
    from research.paraphrase.prompt_bank import PROMPTS

    rows = [{"bank": "design", "bank_id": p["id"], "cluster": p["id"], "text": p["text"],
             "label": p["design_severity"]} for p in PROMPTS]
    rows += [{"bank": "anchored", "bank_id": p["id"], "cluster": p["anchor"], "text": p["text"],
              "label": p["design_severity"]} for p in ANCHORED_PROMPTS]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

_sentence_models: dict[str, object] = {}


def _sentence_model(key: str):
    if key not in _sentence_models:
        from sentence_transformers import SentenceTransformer

        _sentence_models[key] = SentenceTransformer(ENCODERS[key], device="cpu")
    return _sentence_models[key]


def encode(key: str, texts: list[str], cache: bool = True) -> np.ndarray:
    """Normalised sentence embeddings. With cache=True, vectors are stored per text on disk
    (used for training sets); cache=False always runs the encoder (used when timing)."""
    model = _sentence_model(key)
    if not cache:
        return model.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    path = ARTIFACT_DIR / f"embeddings_{key}.npz"
    store: dict[str, np.ndarray] = {}
    if path.exists():
        data = np.load(path, allow_pickle=False)
        store = dict(zip(data["texts"].tolist(), data["vectors"]))
    missing = [t for t in dict.fromkeys(texts) if t not in store]
    if missing:
        vectors = model.encode(missing, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        store.update(zip(missing, vectors))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, texts=np.array(list(store)), vectors=np.stack(list(store.values())))
    return np.stack([store[t] for t in texts])


# ─────────────────────────────────────────────────────────────────────────────
# Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def _logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)


class TfidfLogReg:
    def fit(self, texts, labels, weights):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union

        self.vectorizer = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
        self.clf = _logreg().fit(self.vectorizer.fit_transform(texts), labels, sample_weight=weights)
        return self

    def predict(self, texts, cache: bool = False) -> list[str]:
        return list(self.clf.predict(self.vectorizer.transform(texts)))


class EmbeddingLogReg:
    def __init__(self, key: str):
        self.key = key

    def fit(self, texts, labels, weights):
        self.clf = _logreg().fit(encode(self.key, list(texts)), labels, sample_weight=weights)
        return self

    def predict(self, texts, cache: bool = False) -> list[str]:
        return list(self.clf.predict(encode(self.key, list(texts), cache=cache)))


class FineTunedMiniLM:
    """all-MiniLM-L6-v2 with mean pooling and a linear head, fine-tuned end to end on CPU."""

    def _load(self):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ENCODERS["minilm"])
        self.encoder = AutoModel.from_pretrained(ENCODERS["minilm"])

    def _logits(self, texts):
        batch = self.tokenizer(list(texts), padding=True, truncation=True, max_length=FT_MAX_LEN,
                               return_tensors="pt").to(self.device)
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return self.head((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))

    def fit(self, texts, labels, weights):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(FT_SEED)
        self._load()
        self.device = FT_DEVICE
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        self.encoder.to(self.device)
        self.head.to(self.device)
        texts = list(texts)
        y = torch.tensor([LABELS.index(l) for l in labels], device=self.device)
        counts = torch.bincount(y, minlength=len(LABELS)).float()
        class_weight = len(y) / (len(LABELS) * counts.clamp(min=1))
        w = torch.tensor(np.asarray(weights, dtype=np.float32), device=self.device) * class_weight[y]
        opt = torch.optim.AdamW([{"params": self.encoder.parameters(), "lr": FT_LR_ENCODER},
                                 {"params": self.head.parameters(), "lr": FT_LR_HEAD}])
        gen = torch.Generator().manual_seed(FT_SEED)
        self.encoder.train()
        for _epoch in range(FT_EPOCHS):
            order = torch.randperm(len(texts), generator=gen)
            for start in range(0, len(texts), FT_BATCH):
                idx = order[start:start + FT_BATCH].to(self.device)
                logits = self._logits([texts[i] for i in idx.tolist()])
                loss = (F.cross_entropy(logits, y[idx], reduction="none") * w[idx]).sum() / w[idx].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.encoder.eval()
        self.encoder.to("cpu")          # predictions and latency always on CPU
        self.head.to("cpu")
        self.device = "cpu"
        return self

    def predict(self, texts, cache: bool = False) -> list[str]:
        import torch

        with torch.no_grad():
            return [LABELS[i] for i in self._logits(texts).argmax(-1).tolist()]

    def save(self, path: Path) -> None:
        import torch

        torch.save({"encoder": self.encoder.state_dict(), "head": self.head.state_dict()}, path)

    def load(self, path: Path) -> "FineTunedMiniLM":
        import torch

        self._load()
        self.device = "cpu"
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        state = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(state["encoder"])
        self.head.load_state_dict(state["head"])
        self.encoder.eval()
        return self


def make_classifier(arm: str):
    return {"tfidf_logreg": TfidfLogReg, "minilm_logreg": lambda: EmbeddingLogReg("minilm"),
            "bge_m3_logreg": lambda: EmbeddingLogReg("bge_m3"), "minilm_finetuned": FineTunedMiniLM}[arm]()


# ─────────────────────────────────────────────────────────────────────────────
# Regime R1 (prompt knowledge)
# ─────────────────────────────────────────────────────────────────────────────

_r1_models: dict[str, object] = {}


def _fingerprint(arm: str, train: pd.DataFrame) -> str:
    payload = json.dumps({"arm": arm, "texts": train["text"].tolist(), "labels": train["label"].tolist(),
                          "C": LOGREG_C, "ft": [FT_EPOCHS, FT_LR_ENCODER, FT_LR_HEAD, FT_BATCH, FT_SEED, FT_MAX_LEN]})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prompt_knowledge_model(arm: str):
    """The R1 model for an arm, fitted once and cached on disk."""
    if arm in _r1_models:
        return _r1_models[arm]
    import joblib

    train = template_training_set()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = ARTIFACT_DIR / f"r1_{arm}_{_fingerprint(arm, train)}"
    weights = np.ones(len(train))
    if arm == "minilm_finetuned":
        path = stem.with_suffix(".pt")
        if path.exists():
            model = FineTunedMiniLM().load(path)
        else:
            model = FineTunedMiniLM().fit(train["text"], train["label"], weights)
            model.save(path)
    else:
        # Persist only the sklearn objects: pickling these wrapper classes would tie the file to the
        # module name they were created under (e.g. __main__ when run with -m).
        path = stem.with_suffix(".sklearn.joblib")
        model = make_classifier(arm)
        if path.exists():
            state = joblib.load(path)
            model.clf = state["clf"]
            if "vectorizer" in state:
                model.vectorizer = state["vectorizer"]
        else:
            model.fit(train["text"], train["label"], weights)
            state = {"clf": model.clf}
            if hasattr(model, "vectorizer"):
                state["vectorizer"] = model.vectorizer
            joblib.dump(state, path)
    _r1_models[arm] = model
    return model


def classify_prompt_knowledge(arm: str, text: str) -> str:
    """R1 severity label for one description (no cached embeddings, so latency is real)."""
    return prompt_knowledge_model(arm).predict([text], cache=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Regime R3 (prompt knowledge + every labelled bank prompt), for external text
# ─────────────────────────────────────────────────────────────────────────────

_external_models: dict[str, object] = {}


def external_model(arm: str):
    """R1 sentences plus every labelled prompt of both banks, weighted equally by source.

    No fold is held out: this model is only applied to text from a different source (FMUCD work
    orders), so it answers whether labelled reports from one site transfer to another.
    """
    if arm in _external_models:
        return _external_models[arm]
    import joblib

    templates, labelled = template_training_set(), labelled_bank_prompts()
    texts = templates["text"].tolist() + labelled["text"].tolist()
    labels = templates["label"].tolist() + labelled["label"].tolist()
    weights = np.concatenate([np.ones(len(templates)),
                              np.full(len(labelled), len(templates) / len(labelled))])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"text": texts, "label": labels})
    stem = ARTIFACT_DIR / f"r3_{arm}_{_fingerprint(arm, train)}"
    if arm == "minilm_finetuned":
        path = stem.with_suffix(".pt")
        model = FineTunedMiniLM().load(path) if path.exists() else FineTunedMiniLM().fit(texts, labels, weights)
        if not path.exists():
            model.save(path)
    else:
        path = stem.with_suffix(".sklearn.joblib")
        model = make_classifier(arm)
        if path.exists():
            state = joblib.load(path)
            model.clf = state["clf"]
            if "vectorizer" in state:
                model.vectorizer = state["vectorizer"]
        else:
            model.fit(texts, labels, weights)
            state = {"clf": model.clf}
            if hasattr(model, "vectorizer"):
                state["vectorizer"] = model.vectorizer
            joblib.dump(state, path)
    _external_models[arm] = model
    return model


def classify_external(arm: str, text: str) -> str:
    """R3 severity label for one description."""
    return external_model(arm).predict([text], cache=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Regime R2 (in-domain, cross-validated)
# ─────────────────────────────────────────────────────────────────────────────

def in_domain_predictions(bank: str, arms: list[str] | None = None) -> pd.DataFrame:
    """Cross-validated R2 predictions for every prompt of a bank.

    Anchored bank: leave one anchor out (all five wordings of the anchor are held out).
    Design bank: leave one prompt out. Training = R1 sentences + every labelled prompt of
    both banks outside the held-out cluster, with the in-domain part weighted to equal total weight.
    """
    templates = template_training_set()
    labelled = labelled_bank_prompts()
    test = labelled[labelled["bank"] == bank]
    rows = []
    for arm in arms or ARMS:
        t_arm = time.time()
        for cluster, group in test.groupby("cluster", sort=False):
            held_out = (labelled["bank"] == bank) & (labelled["cluster"] == cluster)
            extra = labelled[~held_out]
            texts = templates["text"].tolist() + extra["text"].tolist()
            labels = templates["label"].tolist() + extra["label"].tolist()
            weights = np.concatenate([np.ones(len(templates)), np.full(len(extra), len(templates) / len(extra))])
            model = make_classifier(arm).fit(texts, labels, weights)
            for r in group.itertuples():
                t0 = time.perf_counter()
                pred = model.predict([r.text], cache=False)[0]
                rows.append({"arm": arm + IN_DOMAIN_SUFFIX, "bank_id": r.bank_id, "severity": pred,
                             "latency_ms": (time.perf_counter() - t0) * 1000, "n_train": len(texts)})
        print(f"[trained_baselines] {bank} {arm}: {test['cluster'].nunique()} folds in {time.time() - t_arm:.0f}s",
              flush=True)
    return pd.DataFrame(rows)


def indomain_path(bank: str) -> Path:
    return RESULTS_DIR / f"indomain_predictions_{bank}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def write_training_summary() -> Path:
    train = template_training_set()
    faults = fault_phrases_from_prompt()
    md = ["# Trained severity classifiers — regime R1 training set\n",
          "**Generated by:** `python -m research.trained_baselines`\n",
          f"Design: `research/AEI/analysis_plan_additions.md` section A. Seed {TEMPLATE_SEED}.\n",
          f"- Fault phrases parsed from the production prompt's FAULT → SENSOR table: {len(faults)} "
          f"({sum(a == 'HIGH' for _, a in faults)} annotated HIGH, {sum(a == 'MEDIUM' for _, a in faults)} annotated MEDIUM)",
          f"- Sentences: {len(train)} after de-duplication; labels: "
          + ", ".join(f"{k} {v}" for k, v in train["label"].value_counts().items()),
          "- Templates: " + "; ".join(f"`{t}`" for t in TEMPLATES + BARE_TEMPLATES),
          "\n## Training accuracy (sanity check, R1 models on their own training set)\n",
          "| Arm | Training accuracy |", "|---|---|"]
    for arm in ARMS:
        model = prompt_knowledge_model(arm)
        preds = model.predict(train["text"].tolist(), cache=True) if arm != "minilm_finetuned" else \
            [p for i in range(0, len(train), 256) for p in model.predict(train["text"].iloc[i:i + 256].tolist())]
        md.append(f"| {arm} | {np.mean(np.array(preds) == train['label'].to_numpy()):.1%} |")
    md.append("\n## Sample sentences\n")
    for label in LABELS:
        md.append(f"- {label}: " + "; ".join(f"\"{t}\"" for t in train[train["label"] == label]["text"].head(4)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "training_set_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", choices=["design", "anchored"], nargs="*", default=[])
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=None)
    parser.add_argument("--no-summary", action="store_true", help="skip fitting R1 models and the training summary")
    parser.add_argument("--device", default="cpu", help="fine-tuning device for minilm_finetuned (cpu or cuda)")
    args = parser.parse_args()
    global FT_DEVICE
    FT_DEVICE = args.device

    if not args.no_summary:
        path = write_training_summary()
        print(f"[trained_baselines] wrote {path.relative_to(PROJECT_ROOT)}")
    for bank in args.bank:
        preds = in_domain_predictions(bank, args.arms)
        out = indomain_path(bank)
        if args.arms and out.exists():
            old = pd.read_csv(out)
            preds = pd.concat([old[~old["arm"].isin(preds["arm"].unique())], preds], ignore_index=True)
        preds.to_csv(out, index=False)
        print(f"[trained_baselines] wrote {out.relative_to(PROJECT_ROOT)} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
