"""Local COCO caption evaluation with separately managed Java resources.

Empty decoded hypotheses remain in the exact selected evaluation set.
This module contains no neural decoder or checkpoint loading and installs
or downloads nothing. Existing Java and local scorer assets are required.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr")


def _image_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value):
        raise ValueError(f"Invalid image or annotation ID: {value!r}")
    return str(value)


def _evaluation_inputs(predictions, labels):
    """Reject partial/duplicate/misaligned IDs; preserve every string hypothesis."""
    if not isinstance(labels, dict) or not labels.get("images") or not labels.get("annotations"):
        raise ValueError("Selected labels must be a COCO dict with images and annotations")
    image_ids = [_image_id(row["id"]) for row in labels["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate selected annotation image IDs")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Predictions must be a nonempty list")
    predicted_ids = [_image_id(row["image_id"]) for row in predictions]
    if len(set(predicted_ids)) != len(predicted_ids):
        raise ValueError("Duplicate prediction image IDs")
    if set(predicted_ids) != set(image_ids):
        raise ValueError("Prediction image IDs must exactly equal the selected annotation image IDs")
    references = {image_id: [] for image_id in image_ids}
    annotation_ids = set()
    for row in labels["annotations"]:
        image_id = _image_id(row["image_id"])
        annotation_id = _image_id(row["id"])
        if annotation_id in annotation_ids:
            raise ValueError("Duplicate caption annotation IDs")
        annotation_ids.add(annotation_id)
        if image_id not in references:
            raise ValueError("Caption annotations contain images outside the selected subset")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError("Reference captions must be nonempty strings")
        references[image_id].append({"caption": row["caption"]})
    if any(not rows for rows in references.values()):
        raise ValueError("Every selected image needs at least one reference caption")
    hypotheses = {}
    for row in predictions:
        if not isinstance(row.get("caption"), str):
            raise ValueError("Predicted captions must be strings; empty hypotheses are preserved")
        hypotheses[_image_id(row["image_id"])] = [{"caption": row["caption"]}]
    # Both mappings use the same order, including official per-image score arrays.
    return references, {image_id: hypotheses[image_id] for image_id in image_ids}


def _caption_diagnostics(predictions):
    """Count empty outputs without rewriting their text or dropping any images."""
    empty = [_image_id(row["image_id"]) for row in predictions if row["caption"] == ""]
    blank = [_image_id(row["image_id"]) for row in predictions if not row["caption"].strip()]
    count = len(predictions)
    return {
        "prediction_count": count,
        "empty_string_count": len(empty),
        "empty_string_image_ids": empty,
        "invalid_caption_definition": "Empty or whitespace-only decoded text; non-string inputs are rejected",
        "invalid_caption_count": len(blank),
        "invalid_caption_rate": len(blank) / count,
        "invalid_caption_image_ids": blank,
        "empty_hypothesis_policy": "Preserve original strings, all images and all references; never substitute words or zero metrics",
    }


def _check_tokenized_inputs(image_ids, reference_counts, references, hypotheses):
    """Fail if tokenization drops empty hypotheses or any selected references."""
    expected = set(image_ids)
    if set(references) != expected or set(hypotheses) != expected:
        raise ValueError("Tokenizer changed reference or prediction IDs")
    for image_id in image_ids:
        if len(references[image_id]) != reference_counts[image_id] or len(hypotheses[image_id]) != 1:
            raise ValueError("Tokenizer changed reference or prediction caption counts")
        if any(not isinstance(value, str) for value in references[image_id] + hypotheses[image_id]):
            raise ValueError("Tokenizer returned a non-string caption")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_scorers(eval_root, output):
    """Copy only existing packages; exclude SPICE and all of its installers."""
    source = Path(eval_root).resolve() / "pycocoevalcap"
    package = output / "scorer_runtime" / "pycocoevalcap"
    if package.exists():
        raise FileExistsError(f"Use a fresh metric output directory: {package}")
    required = ["tokenizer/ptbtokenizer.py", "tokenizer/stanford-corenlp-3.4.1.jar",
                "meteor/meteor.py", "meteor/meteor-1.5.jar", "meteor/data/paraphrase-en.gz",
                "bleu/bleu.py", "rouge/rouge.py", "cider/cider.py"]
    for relative in required:
        if not (source / relative).is_file():
            raise FileNotFoundError(f"Pre-existing metric asset is required; no downloader: {source / relative}")
    # Copy Python source plus the three required runtime assets, not arbitrary
    # scorer caches or unrelated downloaded language resources.
    files = {source / relative for relative in required}
    for name in ("tokenizer", "bleu", "meteor", "rouge", "cider"):
        files.update(path for path in (source / name).rglob("*.py")
                     if "__pycache__" not in path.parts)
    if sum(path.stat().st_size for path in files) > 200 * 1024 * 1024:
        raise RuntimeError("Required scorer runtime exceeds the 200 MiB pilot copy limit")
    package.mkdir(parents=True)
    if (source / "__init__.py").exists():
        shutil.copy2(source / "__init__.py", package / "__init__.py")
    else:
        (package / "__init__.py").touch(exist_ok=False)
    for path in sorted(files):
        target = package / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    assets = {}
    for path in sorted(package.rglob("*")):
        if path.is_file():
            assets[path.relative_to(package).as_posix()] = {"size_bytes": path.stat().st_size,
                                                     "sha256": _sha256(path)}
    return package.parent, assets


def _close_meteor(scorer):
    """Close our worker's METEOR child even when scoring raised while locked."""
    process = getattr(scorer, "meteor_p", None)
    try:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        # Broken Java stdin can fail again on flush/close. Keep
                        # the original scoring error visible during cleanup.
                        pass
    finally:
        lock = getattr(scorer, "lock", None)
        if lock is not None and lock.locked():
            lock.release()


def _metric_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    root = Path(job["scorer_root"]).resolve()
    sys.path.insert(0, str(root))
    modules = {}
    for name in ("tokenizer.ptbtokenizer", "bleu.bleu", "meteor.meteor", "rouge.rouge", "cider.cider"):
        module = importlib.import_module("pycocoevalcap." + name)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Metric import escaped the isolated local scorer directory")
        modules[name] = module
    references, hypotheses = _evaluation_inputs(job["predictions"], job["labels"])
    image_ids = list(references)
    reference_counts = {image_id: len(references[image_id]) for image_id in image_ids}
    diagnostics = _caption_diagnostics(job["predictions"])
    tokenizer = modules["tokenizer.ptbtokenizer"].PTBTokenizer()
    references, hypotheses = tokenizer.tokenize(references), tokenizer.tokenize(hypotheses)
    _check_tokenized_inputs(image_ids, reference_counts, references, hypotheses)
    references = {image_id: references[image_id] for image_id in image_ids}
    hypotheses = {image_id: hypotheses[image_id] for image_id in image_ids}
    tokenized_empty = [image_id for image_id in image_ids if not hypotheses[image_id][0].strip()]
    diagnostics.update({
        "ptb_empty_caption_count": len(tokenized_empty),
        "ptb_empty_caption_rate": len(tokenized_empty) / len(image_ids),
        "ptb_empty_caption_image_ids": tokenized_empty,
    })
    _write_json(job["tokenized_diagnostics_path"], diagnostics)
    meteor = None
    metrics, per_image = {}, {image_id: {} for image_id in image_ids}
    try:
        try:
            meteor = modules["meteor.meteor"].Meteor()
        except Exception as exc:
            raise RuntimeError("METEOR initialization failed; no substitute metrics are emitted") from exc
        scorers = [(modules["bleu.bleu"].Bleu(4), list(METRICS[:4])),
                   (meteor, ["METEOR"]), (modules["rouge.rouge"].Rouge(), ["ROUGE_L"]),
                   (modules["cider.cider"].Cider(), ["CIDEr"])]
        for scorer, names in scorers:
            try:
                aggregate, individual = scorer.compute_score(references, hypotheses)
            except Exception as exc:
                detail = ""
                process = getattr(scorer, "meteor_p", None)
                if process is not None and process.poll() is not None and process.stderr is not None:
                    stderr = process.stderr.read(8192)
                    if stderr:
                        detail = " METEOR stderr: " + stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{'/'.join(names)} scoring failed with {len(tokenized_empty)} empty tokenized hypotheses; "
                    "original empty outputs are preserved and no substitute metrics are emitted" + detail
                ) from exc
            aggregate = aggregate if len(names) > 1 else [aggregate]
            individual = individual if len(names) > 1 else [individual]
            for name, score, values in zip(names, aggregate, individual):
                if not math.isfinite(float(score)) or len(values) != len(image_ids):
                    raise RuntimeError(f"Invalid {name} result")
                metrics[name] = float(score)
                for image_id, value in zip(image_ids, values):
                    value = float(value)
                    if not math.isfinite(value):
                        raise RuntimeError(f"Nonfinite per-image {name} result")
                    per_image[image_id][name] = value
        if set(metrics) != set(METRICS):
            raise RuntimeError("Quick metric set is incomplete")
        result = {"metrics": metrics, "image_count": len(image_ids), "image_ids": image_ids,
                  "caption_diagnostics": diagnostics,
                  "per_image_values": per_image, "scorer_assets": job["scorer_assets"],
                  "java_version": job["java_version"], "protocol": {
                      "tokenizer": "PTBTokenizer", "spice_included": False,
                      "cider_idf_corpus": "the exact fixed selected validation subset only",
                      "cider_comparable_to_full_5000": False,
                      "metric_scale": "raw scorer outputs, not multiplied by 100",
                      "empty_hypotheses": "preserved verbatim; never replaced or excluded",
                      "warning": "Use the identical selected subset for all pilot methods. "
                                 "These are exploratory subset metrics, not full-5000 test results. "
                                 "Per-image BLEU/METEOR means need not equal corpus metrics."}}
        _write_json(job["result_path"], result)
    finally:
        if meteor is not None:
            _close_meteor(meteor)


def eval_metrics(predictions, labels, eval_root, new_outputdir):
    """Return seven raw quick metrics and provenance for an exact selected split.

    Copies the necessary local scorer packages to new_outputdir, probes Java,
    and evaluates in a fresh bounded worker. Neither original scorer assets nor
    original annotations are changed. Output filenames must not already exist.
    """
    _evaluation_inputs(predictions, labels)
    output = Path(new_outputdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "caption_diagnostics.json", _caption_diagnostics(predictions))
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Java is required for existing PTB/METEOR assets; no installation attempted")
    probe = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=15, check=True)
    scorer_root, assets = _prepare_scorers(eval_root, output)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=False)
    result_path, job_path = output / "metrics.json", output / "metric_job.json"
    job = {"predictions": predictions, "labels": labels, "scorer_root": str(scorer_root),
           "scorer_assets": assets, "result_path": str(result_path),
           "tokenized_diagnostics_path": str(output / "tokenized_caption_diagnostics.json"),
           "java_version": (probe.stdout + probe.stderr).strip()}
    _write_json(job_path, job)
    environment = dict(os.environ, TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary),
                       PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--metrics-worker", str(job_path)]
    with (output / "metrics.log").open("x", encoding="utf-8") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=output,
                                  env=environment, start_new_session=(os.name != "nt"))
        try:
            code = worker.wait(timeout=900)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                               stdout=log, stderr=log, timeout=15, check=False)
            else:
                os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=15)
            raise RuntimeError(f"Metric worker exceeded 900 seconds; see {output / 'metrics.log'}")
    if code != 0 or not result_path.is_file():
        raise RuntimeError(f"Quick metrics failed (worker exit {code}); see {output / 'metrics.log'}")
    return json.loads(result_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metrics-worker":
        _metric_worker(sys.argv[2])
    else:
        raise SystemExit("Import eval_metrics; no model decoder, installer or download CLI exists.")
