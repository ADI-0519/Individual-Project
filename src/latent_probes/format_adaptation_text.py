"""CPU-only matched lexical controls for the exposed format-adaptation bank.

This is a new diagnostic instrument, not a modification of the sealed latent
study. Every vocabulary and IDF is fitted on the selected arm's observations.
Only messages reach the text map; formal states provide four fitting labels.
The Offset arm reuses Ahalf256's exact map and coefficients. No language model,
tokenizer, activation cache, test labels, or pickle is needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .format_adaptation_fit import (
    ARMS,
    FACTOR_FIELDS,
    FACTORS,
    fit_intercept_offset,
    fit_lr,
    fold_factors,
    training_indices,
)
from .storage import atomic_json, atomic_npz, canonical_hash, file_sha256, read_json

VIEWS = ("full_messages", "system_only")
FIT_ARMS = ("A512", "Mixed512", "Ahalf256")
SCHEMA = "format-adaptation-text-fit-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def text_policy():
    """The fixed diagnostic policy; no development or held-out selection."""
    return {
        "views": list(VIEWS),
        "arms": list(ARMS),
        "factor_order": list(FACTORS),
        "factor_fields": list(FACTOR_FIELDS),
        "serializer": "exact system/user role and content; newline-separated; no metadata",
        "features": {
            "word": {"analyzer": "word", "ngram_range": [1, 2]},
            "char": {"analyzer": "char", "ngram_range": [3, 5]},
            "common": {
                "lowercase": True,
                "token_pattern": r"(?u)\b\w\w+\b",
                "min_df": 1,
                "max_df": 1.0,
                "max_features": 20000,
                "sublinear_tf": True,
                "norm": "l2",
                "dtype": "float64",
            },
            "combination": "concatenate word and char maps; overall row L2",
            "vocabulary_and_idf": "each arm's selected fitting observations only",
        },
        "C": "512*100/n; lexical feature scale is not the latent feature scale",
        "logistic_regression": {
            "solver": "lbfgs",
            "penalty": "l2",
            "fit_intercept": True,
            "tol": 1e-4,
            "max_iter": 2000,
            "random_state": 11,
            "class_weight": None,
            "sample_weight": None,
        },
        "offset": "Ahalf256 map and coefficients; four intercept updates on B_offset256 only",
        "fold": "1-M1 if B2 else M1 if B1 else I",
        "threshold": "margin > 0; zero predicts 0",
        "thread_limit": 4,
        "failure": "stop on nonconvergence/nonfinite; no rescue, selection, or retry",
        "scope": "lexical diagnostic on an already exposed bank; not a capable semantic judge",
    }


def serialize_messages(messages, view):
    """Serialize content only, with a strict two-message conversation contract."""
    require(view in VIEWS, "Unknown text view")
    require(isinstance(messages, list) and len(messages) == 2, "Need exact system/user messages")
    for message, role in zip(messages, ("system", "user"), strict=True):
        require(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message["role"] == role
            and isinstance(message["content"], str)
            and bool(message["content"].strip()),
            "Messages must contain only the expected role and nonempty content strings",
        )
    selected = messages if view == "full_messages" else messages[:1]
    return "\n".join(f"{message['role']}: {message['content']}" for message in selected)


def _vectorizer(channel, vocabulary=None):
    require(channel in ("word", "char"), "Unknown text channel")
    settings = {
        **text_policy()["features"]["common"],
        **text_policy()["features"][channel],
        "dtype": np.float64,
        "ngram_range": (1, 2) if channel == "word" else (3, 5),
    }
    # sklearn warns when token_pattern is supplied to a character analyzer.
    if channel == "char":
        settings.pop("token_pattern")
    return TfidfVectorizer(vocabulary=vocabulary, **settings)


@dataclass
class TextMap:
    word: TfidfVectorizer
    char: TfidfVectorizer

    def transform(self, texts):
        values = normalize(
            hstack((self.word.transform(texts), self.char.transform(texts)), format="csr"),
            norm="l2",
            copy=False,
        )
        require(values.dtype == np.float64 and np.isfinite(values.data).all(), "Invalid text map")
        return values


def _fit_map(texts):
    mapping = TextMap(_vectorizer("word"), _vectorizer("char"))
    values = normalize(
        hstack(
            (mapping.word.fit_transform(texts), mapping.char.fit_transform(texts)), format="csr"
        ),
        norm="l2",
        copy=False,
    )
    require(values.dtype == np.float64 and np.isfinite(values.data).all(), "Invalid fitted map")
    return mapping, values


def _vocabulary_receipt(vectorizer):
    vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
    return {
        "features": len(vocabulary),
        "vocabulary_sha256": canonical_hash(vocabulary),
        "idf_float64_sha256": sha256(
            np.asarray(vectorizer.idf_, dtype="<f8").tobytes()
        ).hexdigest(),
    }


@dataclass
class TextReadout:
    view: str
    maps: dict
    coefficients: dict
    intercepts: dict
    offsets: np.ndarray
    metadata: dict

    def score(self, records):
        """Score a label-free projection: only messages and renderer are accepted."""
        require(isinstance(records, list) and records, "Need a nonempty list of score records")
        require(
            all(
                isinstance(record, dict)
                and set(record) == {"messages", "renderer"}
                and record["renderer"] in ("A", "B")
                for record in records
            ),
            "Scoring accepts only messages and A/B renderer; labels and metadata are forbidden",
        )
        texts = [serialize_messages(record["messages"], self.view) for record in records]
        shifted = np.array([record["renderer"] == "B" for record in records])
        result = {}
        for arm in FIT_ARMS:
            values = self.maps[arm].transform(texts)
            margins = np.asarray(values @ self.coefficients[arm].T) + self.intercepts[arm]
            require(
                margins.shape == (len(records), 4) and np.isfinite(margins).all(), "Bad margins"
            )
            bits = (margins > 0).astype(np.int64)
            result[arm] = {
                "factor_margins": margins,
                "factor_predictions": bits,
                "verdicts": fold_factors(bits),
            }
        margins = result["Ahalf256"]["factor_margins"].copy()
        margins[shifted] += self.offsets
        require(np.isfinite(margins).all(), "Nonfinite offset margins")
        bits = (margins > 0).astype(np.int64)
        result["Offset"] = {
            "factor_margins": margins,
            "factor_predictions": bits,
            "verdicts": fold_factors(bits),
        }
        return result

    def save(self, output):
        """Write safe JSON/NPZ artifacts into a new directory, never overwrite."""
        output = Path(output)
        require(not output.exists(), "Text fit output already exists")
        output.mkdir(parents=True, exist_ok=False)
        arrays, transforms = {"offsets": self.offsets}, {}
        for arm in FIT_ARMS:
            arrays[arm + "__coefficients"] = self.coefficients[arm]
            arrays[arm + "__intercepts"] = self.intercepts[arm]
            transforms[arm] = {}
            for channel in ("word", "char"):
                vectorizer = getattr(self.maps[arm], channel)
                arrays[arm + "__" + channel + "__idf"] = np.asarray(vectorizer.idf_)
                transforms[arm][channel] = {
                    "vocabulary": {term: int(i) for term, i in vectorizer.vocabulary_.items()}
                }
        atomic_json(
            output / "settings.json",
            {
                "schema": SCHEMA,
                "view": self.view,
                "metadata": self.metadata,
                "transforms": transforms,
            },
        )
        atomic_npz(output / "arrays.npz", **arrays)
        atomic_json(
            output / "manifest.json",
            {
                "schema": "format-adaptation-text-manifest-v1",
                "status": "complete",
                "output_sha256": {
                    name: file_sha256(output / name) for name in ("settings.json", "arrays.npz")
                },
            },
        )


def fit_text(rows, view):
    """Fit all four fixed arms after validating the complete fit-only roster."""
    indices = training_indices(rows)  # Deliberately before text/vocabulary/labels are opened.
    require(view in VIEWS, "Unknown text view")
    texts = [serialize_messages(row.get("messages"), view) for row in rows]
    labels = np.array([[row["state"][field] for field in FACTOR_FIELDS] for row in rows])
    require(labels.shape == (768, 4) and np.isin(labels, (0, 1)).all(), "Invalid factor labels")
    maps, coefficients, intercepts = {}, {}, {}
    metadata = {"schema": SCHEMA, "view": view, "policy": text_policy(), "arms": {}}
    with threadpool_limits(limits=4):
        for arm in FIT_ARMS:
            chosen = indices[arm]
            fit_texts = [texts[i] for i in chosen]
            mapping, values = _fit_map(fit_texts)
            maps[arm] = mapping
            heads = [fit_lr(values, labels[chosen, j], 512 * 100 / len(chosen)) for j in range(4)]
            coefficients[arm] = np.vstack([head.coef_[0] for head in heads])
            intercepts[arm] = np.array([head.intercept_[0] for head in heads])
            metadata["arms"][arm] = {
                "observations": len(chosen),
                "C": 512 * 100 / len(chosen),
                "fit_row_ids": [rows[i]["row_id"] for i in chosen],
                "ordered_serialized_text_sha256": canonical_hash(fit_texts),
                "factor_positive_counts": labels[chosen].sum(axis=0).tolist(),
                "factor_negative_counts": (len(chosen) - labels[chosen].sum(axis=0)).tolist(),
                "word": _vocabulary_receipt(mapping.word),
                "char": _vocabulary_receipt(mapping.char),
            }
        chosen = indices["B_offset"]
        base = (
            maps["Ahalf256"].transform([texts[i] for i in chosen]) @ coefficients["Ahalf256"].T
        ) + intercepts["Ahalf256"]
        fitted = [fit_intercept_offset(base[:, j], labels[chosen, j]) for j in range(4)]
    offsets = np.array([entry[0] for entry in fitted], dtype=np.float64)
    metadata["arms"]["Offset"] = {
        "observations_A": 256,
        "observations_B": len(chosen),
        "feature_map_and_coefficients": "exact Ahalf256 objects",
        "A_fit_row_ids": list(metadata["arms"]["Ahalf256"]["fit_row_ids"]),
        "B_fit_row_ids": [rows[i]["row_id"] for i in chosen],
        "B_ordered_serialized_text_sha256": canonical_hash([texts[i] for i in chosen]),
        "factor_positive_counts_B": labels[chosen].sum(axis=0).tolist(),
        "factor_negative_counts_B": (len(chosen) - labels[chosen].sum(axis=0)).tolist(),
        "factor_offset_optimization": {factor: entry[1] for factor, entry in zip(FACTORS, fitted)},
    }
    return TextReadout(view, maps, coefficients, intercepts, offsets, metadata)


def _load_map(channel, transform, idf, receipt):
    require(isinstance(transform, dict) and set(transform) == {"vocabulary"}, "Bad text transform")
    vocabulary = transform["vocabulary"]
    require(
        isinstance(vocabulary, dict)
        and bool(vocabulary)
        and len(vocabulary) <= 20000
        and all(isinstance(term, str) and term and type(i) is int for term, i in vocabulary.items())
        and set(vocabulary.values()) == set(range(len(vocabulary))),
        "Invalid vocabulary indices",
    )
    require(idf.shape == (len(vocabulary),) and np.all(idf > 0), "Invalid IDF shape or value")
    vectorizer = _vectorizer(channel, vocabulary=vocabulary)
    vectorizer.idf_ = idf.copy()
    require(_vocabulary_receipt(vectorizer) == receipt, "Saved vocabulary/IDF receipt differs")
    return vectorizer


def load_text(directory):
    """Restore the exact saved diagnostic without unpickling or fitting."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    require(
        manifest.get("schema") == "format-adaptation-text-manifest-v1"
        and manifest.get("status") == "complete"
        and set(manifest.get("output_sha256", {})) == {"settings.json", "arrays.npz"},
        "Invalid text artifact manifest",
    )
    for name, digest in manifest["output_sha256"].items():
        require(file_sha256(directory / name) == digest, "Saved text artifact changed: " + name)
    settings = read_json(directory / "settings.json")
    require(settings.get("schema") == SCHEMA and settings.get("view") in VIEWS, "Invalid text fit")
    metadata = settings.get("metadata", {})
    require(
        metadata.get("schema") == SCHEMA
        and metadata.get("policy") == text_policy()
        and metadata.get("view") == settings["view"]
        and set(metadata.get("arms", {})) == set(ARMS)
        and set(settings.get("transforms", {})) == set(FIT_ARMS),
        "Saved text policy differs",
    )
    expected = {"offsets"} | {
        arm + "__" + suffix
        for arm in FIT_ARMS
        for suffix in ("coefficients", "intercepts", "word__idf", "char__idf")
    }
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        require(set(archive.files) == expected, "Unknown or missing saved text arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    require(
        all(a.dtype == np.float64 and np.isfinite(a).all() for a in arrays.values()),
        "Text arrays must be finite float64",
    )
    maps, coefficients, intercepts = {}, {}, {}
    for arm in FIT_ARMS:
        require(set(settings["transforms"][arm]) == {"word", "char"}, "Text channel mismatch")
        mapping = TextMap(
            *(
                _load_map(
                    channel,
                    settings["transforms"][arm][channel],
                    arrays[arm + "__" + channel + "__idf"],
                    metadata["arms"][arm][channel],
                )
                for channel in ("word", "char")
            )
        )
        dimension = len(mapping.word.vocabulary_) + len(mapping.char.vocabulary_)
        coefficients[arm], intercepts[arm] = (
            arrays[arm + "__coefficients"],
            arrays[arm + "__intercepts"],
        )
        require(
            coefficients[arm].shape == (4, dimension) and intercepts[arm].shape == (4,),
            "Saved text head shape differs from vocabulary",
        )
        maps[arm] = mapping
    require(arrays["offsets"].shape == (4,), "Invalid saved factor offsets")
    return TextReadout(
        settings["view"], maps, coefficients, intercepts, arrays["offsets"], copy.deepcopy(metadata)
    )
