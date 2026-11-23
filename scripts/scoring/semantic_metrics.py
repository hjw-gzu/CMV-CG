"""Seven reviewed quick metrics plus optional, unchanged SPICE 1.0.

No downloads, installations, model loads, or source-asset writes. Missing SPICE
never creates a fake zero; all raw predictions/references remain available.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import patch_metrics as quick

SPICE_TIMEOUT = 900
TOKENIZER_TIMEOUT = 180
ADAPTER_SHA256 = "2cdf3190a42082d305e03e5b528585e892a116c9bd45f1af7d40df55872a08b4"
ADAPTER_SOURCE_SHA256 = "a49cca0c01ba97aae4e1922aaf28c95102dc9972a844757e7af14e703f847042"
# Generated from the adjacent SpiceExactOutput.java with javac --release 8.
# Embedded to avoid requiring javac or a separately transferred binary on server.
ADAPTER_BASE64 = "yv66vgAAADQAjAoAAgADBwAEDAAFAAYBABBqYXZhL2xhbmcvT2JqZWN0AQAGPGluaXQ+AQADKClWBwAIAQAiamF2YS9sYW5nL0lsbGVnYWxBcmd1bWVudEV4Y2VwdGlvbggACgEAJ0V4cGVjdGVkIGlucHV0IEpTT04gYW5kIG5ldyBvdXRwdXQgSlNPTgoABwAMDAAFAA0BABUoTGphdmEvbGFuZy9TdHJpbmc7KVYHAA8BABBqYXZhL2xhbmcvU3RyaW5nCgARABIHABMMABQAFQEAE2phdmEvbmlvL2ZpbGUvUGF0aHMBAANnZXQBADsoTGphdmEvbGFuZy9TdHJpbmc7W0xqYXZhL2xhbmcvU3RyaW5nOylMamF2YS9uaW8vZmlsZS9QYXRoOwsAFwAYBwAZDAAaABsBABJqYXZhL25pby9maWxlL1BhdGgBAA50b0Fic29sdXRlUGF0aAEAFigpTGphdmEvbmlvL2ZpbGUvUGF0aDsHAB0BABhqYXZhL25pby9maWxlL0xpbmtPcHRpb24KAB8AIAcAIQwAIgAjAQATamF2YS9uaW8vZmlsZS9GaWxlcwEABmV4aXN0cwEAMihMamF2YS9uaW8vZmlsZS9QYXRoO1tMamF2YS9uaW8vZmlsZS9MaW5rT3B0aW9uOylaBwAlAQAXamF2YS9sYW5nL1N0cmluZ0J1aWxkZXIKACQAAwgAKAEAFlJlZnVzaW5nIHRvIG92ZXJ3cml0ZSAKACQAKgwAKwAsAQAGYXBwZW5kAQAtKExqYXZhL2xhbmcvU3RyaW5nOylMamF2YS9sYW5nL1N0cmluZ0J1aWxkZXI7CgAkAC4MACsALwEALShMamF2YS9sYW5nL09iamVjdDspTGphdmEvbGFuZy9TdHJpbmdCdWlsZGVyOwoAJAAxDAAyADMBAAh0b1N0cmluZwEAFCgpTGphdmEvbGFuZy9TdHJpbmc7BwA1AQAcZWR1L2FudS9zcGljZS9TcGljZUFyZ3VtZW50cwsAFwAxCAA4AQAILXRocmVhZHMIADoBAAEyCAA8AQAHLXN1YnNldAoANAA+DAAFAD8BABYoW0xqYXZhL2xhbmcvU3RyaW5nOylWCQA0AEEMAEIAQwEABWNhY2hlAQASTGphdmEvbGFuZy9TdHJpbmc7CQA0AEUMAEYAQwEACm91dHB1dFBhdGgJADQASAwASQBKAQAHc3luc2V0cwEAE0xqYXZhL2xhbmcvQm9vbGVhbjsKAEwATQcATgwATwBQAQARamF2YS9sYW5nL0Jvb2xlYW4BAAxib29sZWFuVmFsdWUBAAMoKVoHAFIBAB9qYXZhL2xhbmcvSWxsZWdhbFN0YXRlRXhjZXB0aW9uCABUAQAaVW5leHBlY3RlZCBzY29yaW5nIG9wdGlvbnMKAFEADAcAVwEAGWVkdS9hbnUvc3BpY2UvU3BpY2VTY29yZXIKAFYAAwoAVgBaDABbAFwBAApzY29yZUJhdGNoAQAhKExlZHUvYW51L3NwaWNlL1NwaWNlQXJndW1lbnRzOylWCQBWAF4MAF8AYAEABXN0YXRzAQAaTGVkdS9hbnUvc3BpY2UvU3BpY2VTdGF0czsKAGIAYwcAZAwAZQAzAQAYZWR1L2FudS9zcGljZS9TcGljZVN0YXRzAQAMdG9KU09OU3RyaW5nCABnAQABCgkAaQBqBwBrDABsAG0BACFqYXZhL25pby9jaGFyc2V0L1N0YW5kYXJkQ2hhcnNldHMBAAVVVEZfOAEAGkxqYXZhL25pby9jaGFyc2V0L0NoYXJzZXQ7CgAOAG8MAHAAcQEACGdldEJ5dGVzAQAeKExqYXZhL25pby9jaGFyc2V0L0NoYXJzZXQ7KVtCBwBzAQAYamF2YS9uaW8vZmlsZS9PcGVuT3B0aW9uCQB1AHYHAHcMAHgAeQEAIGphdmEvbmlvL2ZpbGUvU3RhbmRhcmRPcGVuT3B0aW9uAQAKQ1JFQVRFX05FVwEAIkxqYXZhL25pby9maWxlL1N0YW5kYXJkT3Blbk9wdGlvbjsJAHUAewwAfAB5AQAFV1JJVEUKAB8AfgwAfwCAAQAFd3JpdGUBAEcoTGphdmEvbmlvL2ZpbGUvUGF0aDtbQltMamF2YS9uaW8vZmlsZS9PcGVuT3B0aW9uOylMamF2YS9uaW8vZmlsZS9QYXRoOwcAggEAHmVkdS9hbnUvc3BpY2UvU3BpY2VFeGFjdE91dHB1dAEABENvZGUBAA9MaW5lTnVtYmVyVGFibGUBAARtYWluAQANU3RhY2tNYXBUYWJsZQEACkV4Y2VwdGlvbnMHAIkBABNqYXZhL2xhbmcvRXhjZXB0aW9uAQAKU291cmNlRmlsZQEAFVNwaWNlRXhhY3RPdXRwdXQuamF2YQAxAIEAAgAAAAAAAgABAAUABgABAIMAAAAdAAEAAQAAAAUqtwABsQAAAAEAhAAAAAYAAQAAAAoACQCFAD8AAgCDAAABOgAHAAQAAADcKr4FnwANuwAHWRIJtwALvyoEMgO9AA64ABC5ABYBAEwrA70AHLgAHpkAHrsAB1m7ACRZtwAmEie2ACkrtgAttgAwtwALv7sANFkHvQAOWQMqAzIDvQAOuAAQuQAWAQC5ADYBAFNZBBI3U1kFEjlTWQYSO1O3AD1NLLQAQMcAFCy0AETHAA0stABHtgBLmgANuwBRWRJTtwBVv7sAVlm3AFhOLSy2AFkruwAkWbcAJi20AF22AGG2ACkSZrYAKbYAMLIAaLYAbgW9AHJZA7IAdFNZBLIAelO4AH1XsQAAAAIAhAAAADYADQAAAAwABgANABAADwAgABAAKwARAEYAEwBXABQAeAAWAJAAFwCaABkAogAaAKcAGwDbAB0AhgAAABAABBD8ADUHABf8AEkHADQJAIcAAAAEAAEAiAABAIoAAAACAIs="


def _new_output(path):
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _asset_inventory(eval_root):
    root = Path(eval_root).resolve() / "pycocoevalcap" / "spice"
    jar = root / "spice-1.0.jar"
    if not jar.is_file():
        raise FileNotFoundError(f"Existing SPICE jar missing; no download: {jar}")
    with zipfile.ZipFile(jar) as archive:
        manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8")
    manifest = manifest.replace("\r\n", "\n").replace("\n ", "")
    fields = {}
    for line in manifest.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    if fields.get("Main-Class") != "edu.anu.spice.SpiceScorer":
        raise ValueError("Unsupported SPICE main class; no automatic metric substitution")
    names = fields.get("Class-Path", "").split()
    if not names or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate SPICE manifest dependencies")
    paths = [jar]
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe/non-local SPICE manifest dependency: {name}")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Existing SPICE dependency missing; no download: {path}")
        paths.append(path)
    required = {"lib/stanford-corenlp-3.6.0.jar", "lib/stanford-corenlp-3.6.0-models.jar",
                "lib/Meteor-1.5.jar", "lib/SceneGraphParser-1.0.jar"}
    if not required.issubset(names):
        raise ValueError("Expected reviewed SPICE 1.0 parser/model dependencies absent")
    return root, jar, paths


def _fingerprints(paths):
    return {str(path): {"size_bytes": path.stat().st_size, "sha256": quick._sha256(path)}
            for path in paths}


def _environment(output):
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=False)
    home = output / "java_home"
    home.mkdir(exist_ok=False)
    env = dict(os.environ, TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env, temporary, home


def _run(command, output, name, env, timeout):
    record = {"command": [str(part) for part in command], "cwd": str(output),
              "timeout_seconds": timeout, "exit_code": None,
              "environment": {key: env[key] for key in ("TEMP", "TMP", "TMPDIR", "PYTHONDONTWRITEBYTECODE")}}
    started = time.monotonic()
    stdout_path, stderr_path = output / (name + ".stdout.log"), output / (name + ".stderr.log")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            child = subprocess.Popen(record["command"], cwd=output, env=env, stdout=stdout,
                                     stderr=stderr, start_new_session=(os.name != "nt"))
            try:
                record["exit_code"] = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   stdout=stdout, stderr=stderr, timeout=15, check=False)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
                record["exit_code"] = child.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            record["spawn_error"] = f"{type(exc).__name__}: {exc}"
    record.update(elapsed_seconds=time.monotonic() - started,
                  stdout_path=str(stdout_path), stderr_path=str(stderr_path))
    quick._write_json(output / (name + ".command.json"), record)
    return record


def _successful(record):
    return record["exit_code"] == 0 and not record.get("timed_out")


def _read_log(path):
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 16384))
        return handle.read().decode("utf-8", errors="replace")


def _javascript_output_failure(record):
    if _successful(record) or record.get("timed_out"):
        return False
    text = _read_log(record["stderr_path"]).lower()
    return (("scriptengine" in text and "nullpointerexception" in text)
            or ("nashorn" in text and ("classnotfound" in text or "noclassdeffound" in text)))


def _materialize_adapter(output):
    data = base64.b64decode(ADAPTER_BASE64, validate=True)
    if hashlib.sha256(data).hexdigest() != ADAPTER_SHA256:
        raise RuntimeError("Bundled output adapter hash mismatch")
    if data[:4] != b"\xca\xfe\xba\xbe" or int.from_bytes(data[6:8], "big") != 52:
        raise RuntimeError("Bundled output adapter is not a Java 8 class")
    root = output / "adapter_classes"
    target = root / "edu" / "anu" / "spice" / "SpiceExactOutput.class"
    target.parent.mkdir(parents=True, exist_ok=False)
    with target.open("xb") as stream:
        stream.write(data)
    return root


def _tokenize_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    root = Path(job["tokenizer_root"]).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("pycocoevalcap.tokenizer.ptbtokenizer")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Tokenizer import escaped isolated copy")
    refs, hyps = quick._evaluation_inputs(job["predictions"], job["labels"])
    ids = list(refs)
    counts = {key: len(value) for key, value in refs.items()}
    tokenizer = module.PTBTokenizer()
    refs, hyps = tokenizer.tokenize(refs), tokenizer.tokenize(hyps)
    quick._check_tokenized_inputs(ids, counts, refs, hyps)
    quick._write_json(job["output_path"],
                      [{"image_id": key, "test": hyps[key][0], "refs": refs[key]} for key in ids])


def _prepare_tokenizer(eval_root, output):
    source = Path(eval_root).resolve() / "pycocoevalcap" / "tokenizer"
    names = ("__init__.py", "ptbtokenizer.py", "stanford-corenlp-3.4.1.jar")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Existing PTB tokenizer asset missing; no download: {source / name}")
    paths = [source / name for name in names]
    if sum(path.stat().st_size for path in paths) > 20 * 1024**2:
        raise RuntimeError("Unexpectedly large PTB tokenizer runtime")
    root = output / "tokenizer_runtime"
    package = root / "pycocoevalcap"
    target = package / "tokenizer"
    target.mkdir(parents=True, exist_ok=False)
    (package / "__init__.py").touch(exist_ok=False)
    for name in names:
        shutil.copy2(source / name, target / name)
        if quick._sha256(source / name) != quick._sha256(target / name):
            raise RuntimeError("Copied tokenizer hash mismatch")
    return root, paths


def _read_spice_result(path, expected_ids):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("SPICE output is not a per-image list")
    ids = [quick._image_id(row["image_id"]) for row in rows]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("SPICE output changed image coverage/order")
    values = {}
    for row, image_id in zip(rows, ids):
        value = row["scores"]["All"]["f"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("SPICE All score is not numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("SPICE All score is nonfinite/out of range")
        values[image_id] = value
    # Match the original Python Spice.compute_score aggregation, retaining all
    # images. Nulls in optional subcategories are allowed by the original scorer.
    import numpy as np
    aggregate = float(np.mean(np.array(list(values.values()), dtype=np.float64)))
    return aggregate, values


def _spice(predictions, labels, eval_root, new_outputdir):
    refs, _ = quick._evaluation_inputs(predictions, labels)
    output = _new_output(new_outputdir)
    quick._write_json(output / "predictions.json", predictions)
    quick._write_json(output / "references.json", labels)
    result = {"status": "unavailable", "reason": None, "image_ids": list(refs),
              "image_count": len(refs), "reference_count": sum(map(len, refs.values())),
              "prediction_path": str(output / "predictions.json"),
              "reference_path": str(output / "references.json"),
              "diagnostics": quick._caption_diagnostics(predictions),
              "method": "Original SPICE 1.0, default synonym matching, no cache; all images/references retained",
              "output_directory": str(output), "commands": [],
              "unavailable_policy": "No SPICE numeric value is emitted; export these predictions/references for local scoring"}
    paths, before = [], None
    try:
        _, jar, paths = _asset_inventory(eval_root)
        spice_paths = list(paths)
        tokenizer_root, token_paths = _prepare_tokenizer(eval_root, output)
        paths = paths + token_paths
        before = _fingerprints(paths)
        quick._write_json(output / "source_hashes_before.json", before)
        java = shutil.which("java")
        if not java:
            raise RuntimeError("Existing Java executable unavailable; no installation attempted")
        env, temporary, home = _environment(output)
        java_flags = [java, "-Xmx4G", "-Djava.io.tmpdir=" + str(temporary),
                      "-Duser.home=" + str(home), "-Dfile.encoding=UTF-8",
                      "-Duser.language=en", "-Duser.country=US"]
        probe = _run([*java_flags, "-version"], output, "java_version", env, 15)
        result["commands"].append(probe)
        if not _successful(probe):
            raise RuntimeError("Existing Java version probe failed")
        result["java_version"] = _read_log(probe["stderr_path"]) + _read_log(probe["stdout_path"])
        input_path = output / "spice_inputs.ptb.json"
        job_path = output / "tokenizer_job.json"
        quick._write_json(job_path, {"predictions": predictions, "labels": labels,
                                   "tokenizer_root": str(tokenizer_root), "output_path": str(input_path)})
        tokenize = _run([sys.executable, "-B", Path(__file__).resolve(), "--tokenize-job", job_path],
                        output, "tokenizer", env, TOKENIZER_TIMEOUT)
        result["commands"].append(tokenize)
        if not _successful(tokenize) or not input_path.is_file():
            raise RuntimeError("Identical PTB tokenizer failed; inspect tokenizer logs")
        tokenized = json.loads(input_path.read_text(encoding="utf-8"))
        if [row["image_id"] for row in tokenized] != list(refs):
            raise RuntimeError("Tokenized SPICE input IDs changed")
        result["ptb_empty_image_ids"] = [row["image_id"] for row in tokenized if not row["test"].strip()]
        original_path = output / "spice_original.json"
        original = _run([*java_flags, "-jar", jar, input_path, "-threads", "2",
                         "-out", original_path, "-subset", "-silent"],
                        output, "spice_original", env, SPICE_TIMEOUT)
        result["commands"].append(original)
        exact_path, backend = original_path, "original_jar_output"
        if not _successful(original):
            if not _javascript_output_failure(original):
                raise RuntimeError("Original SPICE failed for a non-output-compatibility reason; no algorithm fallback")
            adapter_root = _materialize_adapter(output)
            exact_path, backend = output / "spice_adapter.json", "java8_bundled_output_adapter"
            classpath = os.pathsep.join([str(adapter_root), *map(str, spice_paths)])
            adapted = _run([*java_flags, "-cp", classpath, "edu.anu.spice.SpiceExactOutput",
                            input_path, exact_path], output, "spice_adapter", env, SPICE_TIMEOUT)
            result["commands"].append(adapted)
            if not _successful(adapted):
                raise RuntimeError("SPICE output adapter failed; original predictions/references are retained")
            result["adapter"] = {"sha256": ADAPTER_SHA256, "source_sha256": ADAPTER_SOURCE_SHA256,
                                 "class_file_major_version": 52, "server_javac_required": False,
                                 "change": "Only serialize public scorer.stats.toJSONString(); scoring code unchanged"}
        if not exact_path.is_file():
            raise RuntimeError("SPICE returned no exact result JSON")
        aggregate, values = _read_spice_result(exact_path, list(refs))
        result.update(status="available", SPICE=aggregate, per_image_values=values, backend=backend,
                      exact_result_path=str(exact_path), exact_result_sha256=quick._sha256(exact_path),
                      tokenized_input_path=str(input_path), tokenized_input_sha256=quick._sha256(input_path))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError,
            subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if before is not None:
            try:
                after = _fingerprints(paths)
                quick._write_json(output / "source_hashes_after.json", after)
                result["source_assets_unchanged"] = before == after
                result["source_assets"] = before
                if before != after:
                    result["status"], result["reason"] = "unavailable", "Source asset fingerprints changed during scoring"
            except OSError as exc:
                result["status"], result["reason"] = "unavailable", f"Source asset recheck failed: {exc}"
        if result["status"] != "available":
            result.pop("SPICE", None)
            result.pop("per_image_values", None)
        quick._write_json(output / "spice_report.json", result)
    return result


def preflight_spice(eval_root, new_outputdir):
    """Actually test one empty and one identity caption; never install/download.

    Returns status available/unavailable. An unavailable SPICE report is not an
    authorization to modify training; keep predictions/references for later use.
    """
    predictions = [{"image_id": "empty_fixture", "caption": ""},
                   {"image_id": "identity_fixture", "caption": "A cat sits on a mat."}]
    labels = {"images": [{"id": row["image_id"]} for row in predictions],
              "annotations": [{"id": 1, "image_id": "empty_fixture", "caption": "A dog runs."},
                              {"id": 2, "image_id": "identity_fixture", "caption": "A cat sits on a mat."}]}
    result = _spice(predictions, labels, eval_root, new_outputdir)
    if result["status"] == "available":
        expected = {"empty_fixture": 0.0, "identity_fixture": 1.0}
        if result["per_image_values"] != expected:
            result = dict(result, status="unavailable", reason="Empty/identity fixture scores are not the original endpoints")
            result.pop("SPICE", None)
            result.pop("per_image_values", None)
    result["preflight_kind"] = "Actual two-image empty/identity scorer fixture"
    quick._write_json(Path(new_outputdir) / "preflight.json", result)
    return result


def eval_all_metrics(predictions, labels, eval_root, new_outputdir):
    """Return reviewed quick report with real SPICE added, or explicit absence.

    Quick metric failure remains fatal. SPICE absence never masks available
    scores, excludes images, replaces empty text, or emits a fabricated zero.
    """
    quick._evaluation_inputs(predictions, labels)
    output = _new_output(new_outputdir)
    quick._write_json(output / "predictions.json", predictions)
    quick._write_json(output / "references.json", labels)
    report = copy.deepcopy(quick.eval_metrics(predictions, labels, eval_root, output / "quick_metrics"))
    spice = _spice(predictions, labels, eval_root, output / "spice")
    report["spice"] = spice
    report["protocol"]["spice_included"] = spice["status"] == "available"
    report["prediction_path"] = str(output / "predictions.json")
    report["reference_path"] = str(output / "references.json")
    if spice["status"] == "available":
        report["metrics"]["SPICE"] = spice["SPICE"]
        for image_id, value in spice["per_image_values"].items():
            report["per_image_values"][image_id]["SPICE"] = value
    quick._write_json(output / "all_metrics.json", report)
    return report


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--tokenize-job":
        _tokenize_worker(sys.argv[2])
    else:
        raise SystemExit("Import eval_all_metrics/preflight_spice; no downloader or training CLI exists.")

