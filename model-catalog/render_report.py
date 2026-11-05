#!/usr/bin/env python3
"""Render a local, offline lifecycle report without importing the SDK."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


BASE_CHECKS = (
    "1_create_rl_job", "1_sampling_probe", "2_train_two_steps",
    "3_weight_sync", "3_sampling_after_sync", "4_train_two_steps",
    "5_save_checkpoints", "6_cancel_original_job", "7_create_restored_rl_job",
    "7_restored_sampling_probe", "8_train_two_steps", "9_cancel_restored_job",
    "cleanup_delete_training_checkpoint", "cleanup_delete_sampling_checkpoint",
)
STATUSES = ("passed", "failed", "running", "queued", "incomplete")
PHASES = (
    ("1", "Start", "Start"), ("2", "Train 1", "T1"),
    ("3", "Weight sync", "Sync"), ("4", "Train 2", "T2"),
    ("5", "Save checkpoints", "Save"), ("6", "Stop original job", "Stop1"),
    ("7", "Restore checkpoint", "Restore"), ("8", "Train 3", "T3"),
    ("9", "Stop restored job", "Stop2"), ("cleanup", "Cleanup", "Clean"),
)
# Card labels for the Nov 5 catalog. Unknown models keep the pairing id.
SHORT_MODEL_NAMES = {
    "deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
    "gemma-3-4b-it": "Gemma 3 4B",
    "gemma-4-31b-it": "Gemma 4 31B",
    "glm-5.3-flash": "GLM 5.3 Flash",
    "glm-5.3-flash-bf16": "GLM 5.3 Flash BF16",
    "medgemma-1.5-4b-it": "MedGemma 1.5 4B",
    "medgemma-4b-it": "MedGemma 4B",
    "nvidia-nemotron-3-nano-30b-a3b-bf16": "Nemo 3 Nano BF16",
    "nvidia-nemotron-3.5-lightning-30b-a3b-bf16": "Nemo 3.5 Lightning BF16",
    "qwen3-0.6b": "Qwen3 0.6B",
    "qwen3-1.7b": "Qwen3 1.7B",
    "qwen3-30b-a3b": "Qwen3 30B A3B",
    "qwen3-4b-instruct-2507": "Qwen3 4B 2507",
    "qwen3-8b": "Qwen3 8B",
    "qwen3.5-4b": "Qwen3.5 4B",
    "qwen3.5-9b": "Qwen3.5 9B",
    "qwen3.6-35b-a3b": "Qwen3.6 35B A3B",
    "qwen3.8-27b": "Qwen3.8 27B",
    "qwen3.8-flash-next": "Qwen3.8 Flash Next",
}
SECRET_KEYS = {
    "password", "passwd", "pat", "token", "accesstoken", "refreshtoken",
    "authtoken", "apikey", "secret", "clientsecret", "authorization",
    "privatekey", "privatekeypassphrase", "sessiontoken", "cookie",
    "setcookie", "awssecretaccesskey", "awsaccesskeyid", "credentials",
}
SECRET_ASSIGNMENT = re.compile(
    r"""(?i)(["']?(?:password|passwd|pat|access[_-]?token|refresh[_-]?token|"""
    r"""auth[_-]?token|api[_-]?key|client[_-]?secret|secret|session[_-]?token|"""
    r"""aws[_-]?secret[_-]?access[_-]?key|private[_-]?key[_-]?passphrase)["']?"""
    r"""\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,;}&]+)"""
)


def redact_text(value: object) -> str:
    text = str(value)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        "[REDACTED PRIVATE KEY]", text, flags=re.DOTALL,
    )
    text = re.sub(
        r"(?i)(\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*)[^\n]+",
        r"\1[REDACTED]", text,
    )
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [REDACTED]", text)
    text = SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = re.sub(
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
        "[REDACTED JWT]", text,
    )
    return re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)


def redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if re.sub(r"[^a-z0-9]", "", str(key).lower())
            in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def esc(value: object) -> str:
    return html.escape(redact_text(value), quote=True)


def pretty(value: object) -> str:
    return esc(json.dumps(redact(value), indent=2, ensure_ascii=True))


def read_object(path: Path) -> tuple[dict, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}, f"{path.name}: expected a JSON object"
        return value, ""
    except FileNotFoundError:
        return {}, ""
    except (OSError, ValueError) as exc:
        return {}, f"{path.name}: {exc}"


def local_file(root: Path, path: Path) -> bool:
    """Only inspect regular run artifacts, never credential files or outside links."""
    try:
        return (
            path.name.lower() != "qa6.json"
            and path.resolve().name.lower() != "qa6.json"
            and path.resolve().is_relative_to(root.resolve())
            and path.is_file()
        )
    except (OSError, RuntimeError):
        return False


def artifact_link(root: Path, path: Path, label: str | None = None) -> str:
    if not local_file(root, path):
        return ""
    relative = path.relative_to(root).as_posix()
    return f'<a href="{esc(quote(relative, safe="/"))}">{esc(label or relative)}</a>'


def expected_checks(report: dict) -> list[str]:
    names = list(BASE_CHECKS)
    targets = report.get("targets")
    request = report.get("request")
    targets = targets if isinstance(targets, dict) else {}
    request = request if isinstance(request, dict) else {}
    subs = request.get("sub_job_configs")
    subs = subs if isinstance(subs, list) else []
    replay = targets.get("router_replay") is True or any(
        isinstance(sub, dict)
        and isinstance(sub.get("training_config"), dict)
        and isinstance(sub["training_config"].get("router_replay"), dict)
        and sub["training_config"]["router_replay"].get("enabled") is True
        for sub in subs
    )
    if replay:
        names.insert(names.index("1_sampling_probe"), "1_router_replay_bootstrap")
        names.insert(names.index("7_restored_sampling_probe"), "7_router_replay_bootstrap")
    return names


def phase_checks(report: dict) -> list[dict]:
    checks = report.get("checks")
    return [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []


def lifecycle_phases(report: dict) -> list[dict]:
    checks = phase_checks(report)
    recorded = {str(check.get("name", "")) for check in checks}
    expected = expected_checks(report)
    if "1_router_replay_bootstrap" in expected:
        expected += [f"router_replay_generate_step_{step}" for step in range(1, 7)]
    checks = checks + [
        {"name": name, "status": "pending"}
        for name in expected if name not in recorded
    ]
    groups = {key: [] for key, _, _ in PHASES}
    groups["other"] = []
    for check in checks:
        name = str(check.get("name", "Unnamed check"))
        match = re.match(r"^([1-9]|cleanup)_", name)
        key = match[1] if match else "other"
        status = str(check.get("status", "")).lower()
        if status not in ("passed", "failed", "running"):
            status = "pending"
        groups[key].append({"name": name, "status": status})
    phases = []
    for key, name, label in (*PHASES, ("other", "Other", "Other")):
        if not groups[key]:
            continue
        counts = {status: 0 for status in ("passed", "failed", "running", "pending")}
        for check in groups[key]:
            counts[check["status"]] += 1
        status = next(status for status in ("failed", "running", "pending", "passed")
                      if counts[status])
        phases.append({
            "key": key, "name": name, "label": label, "status": status,
            "checks": groups[key], "counts": counts,
        })
    return phases


def phase_tiles(report: dict) -> tuple[str, str]:
    phases = lifecycle_phases(report)
    symbols = {"passed": "&#10003;", "failed": "&#10005;",
               "running": "&#8226;", "pending": "&#183;"}
    total = Counter()
    badges = []
    for phase in phases:
        total.update(phase["counts"])
        label = f'{phase["name"]} | {phase["status"].title()}'
        steps = "".join(
            f'<span class="phase-step" role="img" data-status="{check["status"]}" '
            f'title="{esc(check["name"])} | {check["status"].title()}" '
            f'aria-label="{esc(check["name"])} | {check["status"].title()}">'
            f'{symbols[check["status"]]}</span>' for check in phase["checks"]
        )
        badges.append(
            f'<span class="phase-badge" role="group" data-phase="{phase["key"]}" '
            f'data-status="{phase["status"]}" title="{esc(label)}" aria-label="{esc(label)}">'
            f'<span class="phase-label">{esc(phase["label"])}</span>'
            f'<span class="phase-steps">{steps}</span></span>'
        )
    expected = sum(total.values())
    counts = (
        f'{total["passed"]}/{expected} passed'
        f' \u00b7 {total["failed"]} failed \u00b7 {total["running"]} running'
        f' \u00b7 {total["pending"]} pending'
    )
    focus = next((phase for phase in phases if phase["status"] == "failed"), None)
    prefix = "Stopped"
    if focus is None:
        focus = next((phase for phase in phases if phase["status"] == "running"), None)
        prefix = "Now"
    focus_name = focus["name"] if focus else ""
    if prefix == "Now" and focus_name == "Restore checkpoint":
        focus_name = "Restore ckpt"
    focus_text = f"{prefix}: {focus_name}" if focus else ""
    section = (
        '<span class="tile-phases"><span class="phase-grid">' + "".join(badges) + '</span>'
        '<span class="phase-footer">'
        f'<span class="tile-focus" title="{esc(focus_text)}">{esc(focus_text)}</span>'
        f'<span class="phase-counts" title="{esc(counts)}">{esc(counts)}</span>'
        '</span></span>'
    )
    return section, focus_text


def suite_status(report: dict, ledger: dict, report_error: str = "") -> tuple[str, str]:
    checks = phase_checks(report)
    actual = str(report.get("status", "")).lower()
    scheduled = str(ledger.get("status", "queued")).lower()
    ended = bool(report.get("finished_at") or ledger.get("finished_at")) or (
        actual in ("passed", "failed") or scheduled in ("passed", "failed")
    )
    if ended and ledger.get("cleanup_verified") is False:
        return "failed", "Cleanup verification failed after suite execution."
    if report.get("cleanup_errors"):
        return "failed", "Suite cleanup errors."
    if actual == "failed" or any(check.get("status") == "failed" for check in checks):
        return "failed", str(report.get("error") or "A lifecycle check failed.")
    if scheduled == "failed" and (
        ledger.get("finished_at") or actual not in ("running", "queued")
    ):
        return "failed", str(ledger.get("error") or "Worker execution failed.")
    if actual == "passed":
        by_name = {str(check.get("name")): check.get("status") for check in checks}
        missing = [name for name in expected_checks(report) if by_name.get(name) != "passed"]
        if missing or any(check.get("status") != "passed" for check in checks):
            return "incomplete", "Suite claims passed, but not every required check passed."
        if report.get("active_jobs"):
            return "failed", "Suite reports jobs still active after completion."
        if ledger.get("cleanup_verified") is not True:
            return (
                "incomplete" if scheduled == "passed" else "running",
                "Suite checks passed; awaiting worker cleanup verification.",
            )
        return "passed", ""
    if actual in ("running", "queued"):
        return actual, ""
    if report_error:
        return ("running" if scheduled == "running" else "incomplete"), report_error
    if report:
        return "incomplete", "Suite report has no recognized status."
    if scheduled in ("running", "queued"):
        return scheduled, ""
    return "incomplete", "No suite report; a ledger result alone cannot establish a pass."


def previous_results(root: Path, manifest: dict) -> tuple[dict[str, str], list[str]]:
    previous = manifest.get("previous_run")
    if previous is None:
        return {}, []
    match = re.fullmatch(r"\.\./([A-Za-z0-9][A-Za-z0-9_.-]*)", previous) if isinstance(previous, str) else None
    if not match:
        return {}, ["Previous run must name one sibling directory; comparison not read."]
    root = root.resolve()
    baseline = root.parent / match[1]
    if baseline == root:
        return {}, ["Previous run must be different from the current run; comparison not read."]
    try:
        if baseline.resolve() != baseline:
            return {}, ["Previous run resolves outside the named sibling; comparison not read."]
    except (OSError, RuntimeError):
        return {}, ["Previous run path could not be verified; comparison not read."]
    documents, notices = {}, []
    for name in ("manifest.json", "ledger.json"):
        path = baseline / name
        if not local_file(baseline, path):
            documents[name] = {}
            continue
        documents[name], error = read_object(path)
        if error:
            notices.append(f"Previous run {error}")
    pairings = documents["manifest.json"].get("pairings")
    entries = documents["ledger.json"].get("entries")
    entries = entries if isinstance(entries, dict) else {}
    results = {}
    for pairing in pairings if isinstance(pairings, list) else []:
        if not isinstance(pairing, dict):
            continue
        identifier = str(pairing.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in (".", ".."):
            continue
        report, error = {}, ""
        path = baseline / "configs" / identifier / "report.json"
        if local_file(baseline, path):
            report, error = read_object(path)
        entry = entries.get(identifier)
        results[identifier] = suite_status(
            report, entry if isinstance(entry, dict) else {}, error,
        )[0]
    return results, notices


def result_change(status: str, previous: str | None) -> str:
    if status not in ("passed", "failed") or previous not in ("passed", "failed"):
        return ""
    if status == previous:
        return "Unchanged"
    return "Fixed" if status == "passed" else "Regressed"


def badge(status: object) -> str:
    status = str(status)
    style = status if status in STATUSES else "incomplete"
    symbols = {"passed": "&#10003;", "failed": "&#10007;", "running": "&#8226;",
               "queued": "&#183;", "incomplete": "!"}
    return (
        f'<span class="status {style}"><span aria-hidden="true">{symbols[style]}</span>'
        f" {esc(status.replace('_', ' ').title())}</span>"
    )


def config_tile(
    pairing: dict, status: str, anchor: str, search_text: str, runtime: str | None = "Not started",
    report: dict | None = None,
) -> str:
    model = str(pairing.get("model", "Unknown"))
    context = str(pairing.get("context", "Unknown"))
    mode = str(pairing.get("mode", "Unknown"))
    short = SHORT_MODEL_NAMES.get(model, model)
    compact_context = "std" if context == "standard" else context
    compact_mode = "LoRA" if mode == "lora" else mode
    label = f"{model} | {context} / {mode} | {status.title()}"
    if runtime is not None:
        label += f" | E2E {runtime}"
    search_text = f"{search_text} {short} {compact_context} / {compact_mode} {label}"
    symbol = {"passed": "&#10003;", "failed": "&#10005;", "running": "&#8226;",
              "queued": "&#183;", "incomplete": "!"}[status]
    phases, _ = phase_tiles(report or {})
    runtime_html = (
        f'<span class="tile-runtime">E2E {esc(runtime)}</span>' if runtime is not None else ""
    )
    tile_class = "config-tile has-runtime" if runtime is not None else "config-tile"
    return (
        f'<a class="{tile_class}" href="#{anchor}" data-config-link data-status="{status}" '
        f'data-model="{esc(model)}" data-detail="{anchor}" data-id="{esc(str(pairing.get("id", "")))}" '
        f'data-search="{esc(search_text.lower())}" '
        f'title="{esc(label)}" aria-label="{esc(label)}">'
        f'<span class="tile-model">{esc(short)}</span>'
        f'<span class="tile-config">{esc(compact_context)} / {esc(compact_mode)}</span>'
        f'{runtime_html}'
        f'<span class="tile-symbol" aria-hidden="true">{symbol}</span>'
        f'{phases}</a>'
    )


def elapsed(start: object, finish: object, now: datetime) -> str:
    if not start:
        return "Not started"
    try:
        first = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        last = datetime.fromisoformat(str(finish).replace("Z", "+00:00")) if finish else now
        if first.tzinfo is None or last.tzinfo is None:
            return "Unknown"
        seconds = int((last - first).total_seconds())
        if seconds < 0:
            return "Unknown"
        hours, rest = divmod(seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "Unknown"


def runtime_bounds(report: dict, entry: dict) -> tuple[object, object]:
    if entry.get("status") == "queued":
        return None, None
    finish = entry.get("finished_at")
    if not finish and not (entry.get("started_at") and entry.get("status") == "running"):
        finish = report.get("finished_at")
    return entry.get("started_at") or report.get("started_at"), finish


def config_runtime(report: dict, entry: dict, now: datetime) -> str:
    start, finish = runtime_bounds(report, entry)
    return elapsed(start, finish, now)


def readable_jsonl(text: str) -> str:
    records = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            records.append(line)
            continue
        if not isinstance(value, dict):
            records.append(str(value))
            continue
        value = redact(value)
        used = set()
        prefix = []
        for keys in (("timestamp", "time", "ts"), ("level", "severity", "levelname")):
            for key in keys:
                if key in value:
                    prefix.append(str(value[key]))
                    used.add(key)
                    break
        messages = []
        for key in ("message", "msg", "text", "log", "exception", "exc_info",
                    "traceback", "error", "stacktrace"):
            if key in value:
                item = value[key]
                messages.append(item if isinstance(item, str) else json.dumps(item, indent=2))
                used.add(key)
        rest = {key: item for key, item in value.items() if key not in used}
        body = "\n".join(messages)
        if rest:
            body += ("\n" if body else "") + json.dumps(rest, indent=2)
        records.append((" ".join(prefix) + " " + body).strip())
    return "\n".join(records)


def error_excerpt(root: Path, path: Path) -> str:
    if not local_file(root, path):
        return ""
    try:
        with path.open("rb") as handle:
            head = handle.read(65536)
            handle.seek(0, 2)
            size = handle.tell()
            if size > 65536:
                handle.seek(max(65536, size - 65536))
                head += b"\n[...]\n" + handle.read(65536)
        text = head.decode("utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\x00", "")
        if path.suffix.lower() == ".jsonl":
            text = readable_jsonl(text)
        text = redact_text(text)
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if re.search(
            r"error|exception|traceback|fail|invalid_config|out of memory|timeout",
            line, flags=re.IGNORECASE,
        )]
        selected = set()
        for index in hits[-20:]:
            selected.update(range(max(0, index - 2), min(len(lines), index + 4)))
        if not selected:
            selected.update(range(max(0, len(lines) - 16), len(lines)))
        return "\n".join(lines[index] for index in sorted(selected))[:20000]
    except OSError as exc:
        return f"Unable to read log excerpt: {exc}"


def field(label: str, value: object) -> str:
    return f"<dt>{esc(label)}</dt><dd>{esc(value if value is not None else 'Not recorded')}</dd>"


def pairing_anchor(identifier: str, index: int, used: set[str]) -> str:
    slug = identifier if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", identifier) else ""
    if not slug:
        slug = f"config-{index + 1}"
    if slug in used:
        slug = f"{slug}-{index + 1}"
    used.add(slug)
    return slug


def one_line(value: object, limit: int = 400) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        text = json.dumps(redact(value), ensure_ascii=True)
    else:
        text = redact_text(str(value))
    text = re.sub(r"\s+", " ", text).strip() or "-"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def first_failure(report: dict, status: str, reason: str) -> tuple[str, str]:
    failed = [check for check in phase_checks(report) if check.get("status") == "failed"]
    if failed:
        check = failed[0]
        name = str(check.get("name") or "-")
        if len(failed) > 1:
            name = f"{name} (+{len(failed) - 1} more)"
        return name, one_line(check.get("error") or report.get("error") or reason)
    if status == "passed":
        return "-", "-"
    return "-", one_line(report.get("error") or reason)


def debug_packet(
    identifier: str, status: str, report: dict, reason: str, directory: Path | None, root: Path,
) -> str:
    failed_check, error = first_failure(report, status, reason)
    paths = []
    if directory is not None:
        for name in ("report.json", "console.log"):
            path = directory / name
            if local_file(root, path):
                paths.append(path.relative_to(root).as_posix())
    packet = "\n".join((
        f"id: {identifier or '-'}",
        f"status: {status}",
        f"failed_check: {failed_check}",
        f"error: {error}",
        f"artifacts: {' '.join(paths) or '-'}",
        f"rerun: PAIRINGS={identifier}" if identifier else "rerun: -",
    ))
    return f'<pre class="debug-packet">{esc(packet)}</pre>'


def failure_item(pairing: dict, status: str, anchor: str, report: dict, reason: str) -> str:
    identifier = str(pairing.get("id", "") or anchor)
    failed_check, error = first_failure(report, status, reason)
    return (
        f'<a class="failure-item" href="#{anchor}" data-config-link data-detail="{anchor}">'
        f'<span class="failure-id">{esc(identifier)}</span>'
        f'<span class="failure-check">{esc(failed_check)}</span>'
        f'<span class="failure-error">{esc(one_line(error, 180))}</span></a>'
    )


def cleanup_label(report: dict, ledger: dict) -> str:
    if ledger.get("cleanup_verified") is True:
        return "Verified"
    if ledger.get("cleanup_verified") is False:
        return "Not verified"
    checks = phase_checks(report)
    if report.get("cleanup_errors") or any(
        check.get("status") == "failed" and str(check.get("name", "")).startswith("cleanup_")
        for check in checks
    ):
        return "Failed"
    return "Not recorded"


def config_details(root: Path, pairing: dict, ledger: dict, report: dict,
                   status: str, reason: str, report_error: str, anchor: str,
                   directory: Path | None, now: datetime, image_tag: object = None,
                   previous: str | None = None, change: str = "") -> str:
    checks = phase_checks(report)
    phases = []
    for check in checks:
        seconds = check.get("seconds")
        duration = f"{seconds} s" if seconds is not None else "Not recorded"
        if seconds is None and check.get("status") == "running":
            duration = "In progress"
        result = ""
        if check.get("result") is not None:
            result = f"<details><summary>Result</summary><pre>{pretty(check['result'])}</pre></details>"
        phases.append(
            f"<tr><th scope=\"row\">{esc(check.get('name', 'Unnamed check'))}</th>"
            f"<td>{badge(check.get('status', 'unknown'))}</td><td>{esc(duration)}</td>"
            f"<td>{esc(check.get('error', ''))}{result}</td></tr>"
        )
    present = {str(check.get("name")) for check in checks}
    expected = [check["name"] for phase in lifecycle_phases(report) for check in phase["checks"]]
    for name in expected:
        if name not in present:
            phases.append(
                f'<tr><th scope="row">{esc(name)}</th><td>{badge("not_recorded")}</td>'
                '<td>Not recorded</td><td></td></tr>'
            )
    errors = []
    for label, value in (
        ("Status evidence", reason), ("Suite error", report.get("error")),
        ("Cleanup errors", report.get("cleanup_errors")),
        ("Ledger error", ledger.get("error")), ("Report read error", report_error),
    ):
        if value:
            content = pretty(value) if isinstance(value, (dict, list)) else esc(value)
            errors.append(f"<h4>{esc(label)}</h4><pre>{content}</pre>")
    links, excerpts = [], []
    image_html = '<p class="muted">Image metadata proof not yet recorded.</p>'
    if directory is not None:
        audit_path = directory / "image-audit.json"
        if local_file(root, audit_path):
            audit, audit_error = read_object(audit_path)
            image_html = (
                f'<pre>{esc(audit_error) if audit_error else pretty(audit)}</pre>'
            )
        for path, label in (
            (directory / "report.json", "Report JSON"),
            (audit_path, "Image audit JSON"),
            (directory / "console.log", "Console log"),
            (directory.parent / f"{directory.name}.console.log", "Live console log"),
        ):
            link = artifact_link(root, path, label)
            if link:
                links.append(link)
                if status == "failed" and path.name.endswith(".log"):
                    excerpts.append(f"<h4>{esc(label)}</h4><pre>{esc(error_excerpt(root, path))}</pre>")
        execution = directory / "execution-logs"
        if execution.resolve().is_relative_to(root) and execution.is_dir():
            for path in sorted(execution.rglob("*")):
                link = artifact_link(root, path)
                if link:
                    links.append(link)
                    if status == "failed" and path.suffix.lower() in (".log", ".txt", ".out", ".err", ".json", ".jsonl", ""):
                        excerpt = error_excerpt(root, path)
                        if excerpt:
                            excerpts.append(f"<h4>{esc(path.relative_to(root))}</h4><pre>{esc(excerpt)}</pre>")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if relative.parts[0] == "execution-logs" or relative.as_posix() in (
                "report.json", "console.log", "image-audit.json",
            ):
                continue
            link = artifact_link(root, path)
            if link:
                links.append(link)
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if isinstance(diagnostic, dict) and diagnostic.get("error"):
                errors.append(f"<h4>Diagnostic collection error</h4><pre>{esc(diagnostic['error'])}</pre>")
    start, finish = runtime_bounds(report, ledger)
    times = "".join(field(label, value) for label, value in (
        ("Suite status", report.get("status", "No report")),
        ("Ledger status", ledger.get("status", "No entry")),
        ("Worker", ledger.get("worker")), ("Return code", ledger.get("returncode")),
        ("Started", start),
        ("Finished", finish),
        ("Elapsed", config_runtime(report, ledger, now)),
        ("Cleanup verification", cleanup_label(report, ledger)),
        ("Training config", pairing.get("training_config")),
        ("Sampling config", pairing.get("inference_config")),
        ("Requested image tag", image_tag or "Deployment default"),
        ("Previous result", previous.title() if previous else "Not recorded"),
        ("Change", change or "Not yet comparable"),
    ))
    request = report.get("request", pairing.get("request"))
    request_html = (
        f"<pre>{pretty(request)}</pre>" if request is not None
        else '<p class="muted">Not recorded</p>'
    )
    if report.get("_request_from_archive"):
        request_html = (
            '<p class="muted">Archived composed request; no executed request recorded.</p>'
            + request_html
        )
    job_html = pretty({
        "cleanup_verified": ledger.get("cleanup_verified"),
        "job_statuses": ledger.get("job_statuses"),
        "active_jobs": report.get("active_jobs"),
        "checkpoints": report.get("checkpoints"),
    })
    return (
        f'<article class="config-detail" id="{anchor}">'
        f'<div class="detail-heading">{badge(status)}'
        f"<span>{esc(pairing.get('id', 'Missing ID'))}</span>"
        '<a class="muted" href="#results">Back to cards</a></div>'
        + debug_packet(str(pairing.get("id", "")), status, report, reason, directory, root)
        + '<div class="detail-body">'
        + "".join(errors) + f"<dl>{times}</dl>"
        '<h3>Phases</h3><div class="table-scroll"><table class="phases">'
        '<thead><tr><th>Check</th><th>Status</th><th>Duration</th><th>Error / result</th></tr></thead>'
        f"<tbody>{''.join(phases)}</tbody></table></div>"
        f'<h3>Image metadata proof</h3>{image_html}'
        f'<h3>Cleanup / jobs</h3><pre>{job_html}</pre>'
        f"<h3>Request</h3>{request_html}"
        "<h3>Artifacts</h3>" + (
            '<ul class="artifacts">' + "".join(f"<li>{link}</li>" for link in links) + "</ul>"
            if links else '<p class="muted">None recorded</p>'
        ) + ("<h3>Error excerpts</h3>" + "".join(excerpts) if excerpts else "")
        + "</div></article>"
    )


CSS = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #202628; background: #f6f7f7; }
* { box-sizing: border-box; letter-spacing: 0; }
body { margin: 0; font-size: 13px; line-height: 1.45; }
main { width: 100%; margin: auto; padding: 20px 24px 48px; }
h1 { font-size: 23px; margin: 0; } h2 { font-size: 17px; margin: 22px 0 8px; }
h3 { font-size: 14px; margin: 20px 0 6px; } h4 { font-size: 13px; margin: 10px 0 4px; }
h1, h2, h3, h4 { overflow-wrap: anywhere; }
header { display: flex; align-items: start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.report-top { display: contents; }
p { margin: 5px 0; } .muted { color: #606b70; }
a { color: #075b92; text-underline-offset: 2px; overflow-wrap: anywhere; }
a:hover { color: #003b61; } :focus-visible { outline: 3px solid #087da5; outline-offset: 3px; }
.summary, .metrics { display: flex; flex-wrap: wrap; gap: 12px 22px; padding: 12px 0; border-bottom: 1px solid #ccd3d5; }
.summary strong { font-size: 19px; margin-right: 5px; } .metrics strong { margin-left: 6px; }
.image-tag { font-size: 16px; margin: 10px 0; overflow-wrap: anywhere; }
.comparison { padding: 10px 0; border-bottom: 1px solid #ccd3d5; overflow-wrap: anywhere; }
.comparison-counts { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.fixed { color: #166534; } .regressed { color: #b91c1c; }
.controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin: 16px 0; }
label { display: grid; gap: 4px; font-weight: 600; }
input, select { font: inherit; background: white; border: 1px solid #a5b2b7; border-radius: 4px; padding: 7px 9px; min-height: 35px; max-width: 100%; }
.view-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 8px 0; }
input { width: 310px; } select { min-width: 145px; } #visible-count { align-self: center; }
.table-scroll { overflow-x: auto; border: 1px solid #cdd5d8; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 12px; }
thead { background: #e9eeef; } th, td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid #dfe4e6; }
thead th { white-space: nowrap; font-weight: 650; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 5px; }
.config-tile { display: grid; grid-template-columns: minmax(0, 1fr) 16px;
grid-template-rows: auto auto minmax(0, 1fr); min-height: 116px; min-width: 0;
gap: 2px 4px; padding: 5px 6px; border-radius: 5px;
background: #e5eaed; color: #35474e; box-shadow: inset 0 0 0 1px #c2ccd0; font-size: 11px; line-height: 12px; }
.config-tile.has-runtime { grid-template-rows: auto auto auto minmax(0, 1fr); min-height: 128px; }
.config-tile, .config-tile:hover { text-decoration: none; }
.config-tile:hover { filter: brightness(.94); color: #35474e; }
.config-tile[data-status="passed"] { background: #166534; box-shadow: inset 0 0 0 1px #166534; color: #fff; }
.config-tile[data-status="failed"] { background: #b91c1c; box-shadow: inset 0 0 0 1px #b91c1c; color: #fff; }
.config-tile[data-status="running"] { background: #075985; box-shadow: inset 0 0 0 1px #075985; color: #fff; }
.config-tile[data-status="queued"] { background: #59636a; box-shadow: inset 0 0 0 1px #59636a; color: #fff; }
.config-tile[data-status="incomplete"] { background: #855400; box-shadow: inset 0 0 0 1px #855400; color: #fff; }
.config-tile.selected { box-shadow: inset 0 0 0 2px #fff, 0 0 0 2px #202628; }
.tile-model { grid-column: 1; grid-row: 1; font-weight: 650; font-size: 11px; line-height: 12px; }
.tile-model, .tile-config, .tile-runtime, .tile-focus, .phase-label { min-width: 0; min-height: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tile-config { grid-column: 1; grid-row: 2; }
.tile-runtime { grid-column: 1; grid-row: 3; font-size: 11px; line-height: 12px; font-variant-numeric: tabular-nums; }
.tile-symbol { grid-column: 2; grid-row: 1 / 3; align-self: center; justify-self: center; font-size: 18px; }
.has-runtime .tile-symbol { grid-row: 1 / 4; }
.tile-phases { grid-column: 1 / -1; grid-row: 3; display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 3px; min-width: 0; min-height: 0; }
.has-runtime .tile-phases { grid-row: 4; }
.phase-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); grid-auto-rows: minmax(26px, 1fr); gap: 2px; min-width: 0; min-height: 0; }
.phase-badge { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1px; padding: 1px;
min-width: 0; min-height: 0; border-radius: 3px; box-shadow: inset 0 0 0 1px #ffffff80; }
.phase-label { max-width: 100%; font-size: 8px; line-height: 9px; font-weight: 650; }
.phase-steps { display: flex; flex-wrap: wrap; justify-content: center; gap: 0 2px; min-width: 0; min-height: 0; }
.phase-step { font-size: 8px; line-height: 9px; }
.phase-badge[data-status="passed"] { background: #166534; color: #fff; }
.phase-badge[data-status="failed"] { background: #b91c1c; color: #fff; }
.phase-badge[data-status="running"] { background: #075985; color: #fff; }
.phase-badge[data-status="pending"] { background: #e5eaed; color: #35474e; }
.phase-footer { display: grid; gap: 1px; min-width: 0; min-height: 0; }
.tile-focus { font-size: 10px; line-height: 11px; }
.phase-counts { font-size: 9px; line-height: 10px; overflow-wrap: anywhere; }
.status { display: inline-flex; align-items: center; gap: 4px; font-weight: 650; white-space: nowrap; }
.status > span { display: inline-grid; place-items: center; width: 15px; height: 15px; }
.passed { color: #166534; } .failed { color: #b91c1c; } .running { color: #075985; }
.queued { color: #59636a; } .incomplete { color: #855400; }
.config-detail { display: none; margin-top: 18px; padding: 14px 16px 8px; border: 1px solid #cdd5d8; border-radius: 8px; background: #fff; scroll-margin-top: 12px; }
.config-detail:target { display: block; }
.config-detail > .detail-heading { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px; margin: 0 0 10px; }
.config-detail > .detail-heading .muted { margin-left: auto; }
.detail-body { padding: 0 0 8px; }
dl { display: grid; grid-template-columns: 160px minmax(0, 1fr); gap: 6px 16px; }
dt { color: #606b70; } dd { margin: 0; overflow-wrap: anywhere; }
pre { margin: 4px 0 12px; padding: 10px 12px; background: #edf0f1; border-left: 3px solid #aab8be; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; max-height: 420px; overflow: auto; }
.debug-packet { margin: 0 0 12px; background: #202628; color: #e8eeef; border-left: 0; max-height: none; overflow: visible; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.phases { min-width: 730px; } .phases th { max-width: 350px; overflow-wrap: anywhere; }
.phases td:last-child { max-width: 650px; overflow-wrap: anywhere; }
.artifacts { margin: 0; padding-left: 20px; }
.notice { border-left: 3px solid #b57c19; padding: 8px 12px; background: #fff5d8; }
.failure-text { color: #a31a1a; } [hidden] { display: none !important; }
.failure-rail { background: #fff; border: 1px solid #cdd5d8; border-radius: 8px; padding: 10px; min-width: 0; }
.failure-rail h2 { margin: 0 0 8px; font-size: 14px; }
.failure-rail h2 span { font-weight: 400; color: #606b70; }
.failure-list { display: grid; gap: 6px; }
.failure-item { display: grid; gap: 2px; padding: 8px 9px; border-radius: 5px;
background: #f8eaea; box-shadow: inset 3px 0 0 #b91c1c; color: #202628; text-decoration: none; }
.failure-item:hover { color: #202628; filter: brightness(.97); }
.failure-item.selected { box-shadow: inset 3px 0 0 #b91c1c, 0 0 0 2px #202628; }
.failure-id { font-weight: 650; font-size: 12px; overflow-wrap: anywhere; }
.failure-check { color: #b91c1c; font-weight: 650; font-size: 11px; }
.failure-error { color: #35474e; font-size: 11px; line-height: 14px;
display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.report-body { min-width: 0; }
@media (min-width: 900px) {
main.has-failures { display: grid; grid-template-columns: minmax(240px, 300px) minmax(0, 1fr); gap: 16px 20px; align-items: start; }
main.has-failures .failure-rail { position: sticky; top: 12px; max-height: calc(100dvh - 24px); overflow: auto; }
}
@media (min-width: 900px) {
main { padding: 12px 20px 40px; } header h1 { font-size: 21px; } header p { margin: 2px 0; }
.image-tag { display: inline-block; font-size: 14px; margin: 4px 18px 4px 0; }
.summary, .metrics { padding: 6px 0; } .summary strong { font-size: 18px; }
.comparison { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; padding: 6px 0; }
.comparison p { margin: 0; } .controls { margin: 10px 0; gap: 10px; }
}
@media (max-width: 600px) { main { padding: 14px 12px 32px; } header { display: block; }
input { width: 100%; } .controls > label:first-child { width: 100%; }
.config-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
dl { grid-template-columns: 110px minmax(0, 1fr); gap: 6px 8px; }
}
"""

JS = """
(() => {
  const search = document.getElementById('search');
  const status = document.getElementById('status-filter');
  const model = document.getElementById('model-filter');
  const tiles = [...document.querySelectorAll('.config-tile')];
  const failures = [...document.querySelectorAll('.failure-item')];
  function markSelected() {
    const id = location.hash.slice(1);
    tiles.forEach(tile => tile.classList.toggle('selected', tile.dataset.detail === id));
    failures.forEach(item => item.classList.toggle('selected', item.dataset.detail === id));
  }
  function filter() {
    let visible = 0;
    const query = search.value.trim().toLowerCase();
    tiles.forEach(tile => {
      const show = (!query || tile.dataset.search.includes(query))
        && (!status.value || tile.dataset.status === status.value)
        && (!model.value || tile.dataset.model === model.value);
      tile.hidden = !show;
      if (show) visible++;
    });
    document.getElementById('visible-count').textContent = visible + ' / ' + tiles.length + ' configs';
    document.getElementById('no-results').hidden = visible !== 0;
  }
  [search, status, model].forEach(control => control.addEventListener('input', filter));
  function openHash() {
    const target = document.getElementById(location.hash.slice(1));
    markSelected();
    if (!target || !target.classList.contains('config-detail')) return;
    target.scrollIntoView({block: 'start'});
  }
  window.addEventListener('hashchange', openHash);
  filter();
  openHash();
})();
"""


def pacific_time(value: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        local = value.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:
        local = value.astimezone(timezone(timedelta(hours=-8), "PST"))
    return local.strftime("%Y-%m-%d %H:%M:%S ") + local.tzname()


def snapshot_time(ledger: dict, *, terminal: bool) -> datetime:
    if terminal:
        try:
            finished = datetime.fromisoformat(str(ledger.get("finished_at")).replace("Z", "+00:00"))
            if finished.tzinfo is not None:
                return finished.astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


def render(root: Path) -> str:
    root = root.resolve()
    documents, notices = {}, []
    for name in ("manifest.json", "ledger.json"):
        path = root / name
        if local_file(root, path):
            documents[name], error = read_object(path)
            if error:
                notices.append(error)
        else:
            documents[name] = {}
            if path.exists() or path.is_symlink():
                notices.append(f"{name}: unsafe artifact; not read.")
    manifest, ledger = documents["manifest.json"], documents["ledger.json"]
    pairings = manifest.get("pairings", [])
    if not (root / "manifest.json").is_file():
        notices.append("Manifest not yet available.")
    if not isinstance(pairings, list):
        pairings = []
        notices.append("Manifest pairings must be a list.")
    valid = [item for item in pairings if isinstance(item, dict)]
    if len(valid) != len(pairings):
        notices.append("Some manifest pairings are not JSON objects.")
    entries = ledger.get("entries")
    entries = entries if isinstance(entries, dict) else {}
    previous, previous_notices = previous_results(root, manifest)
    current_ids = {str(pair.get("id", "")) for pair in valid}
    previous = {identifier: status for identifier, status in previous.items()
                if identifier in current_ids}
    notices.extend(previous_notices)
    counts, changes, tile_inputs, detail_inputs, failure_items = Counter(), Counter(), [], [], []
    models = sorted({str(pair.get("model", "Unknown")) for pair in valid})
    used_anchors: set[str] = set()
    for index, pairing in enumerate(sorted(valid, key=lambda p: (
        str(p.get("model", "")), str(p.get("context", "")), str(p.get("mode", "")),
    ))):
        identifier = str(pairing.get("id", ""))
        entry = entries.get(identifier, {})
        entry = entry if isinstance(entry, dict) else {}
        directory = None
        report, report_error = {}, ""
        if re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) and identifier not in (".", ".."):
            candidate = root / "configs" / identifier
            if candidate.resolve().is_relative_to(root):
                directory = candidate
                if local_file(root, directory / "report.json"):
                    report, report_error = read_object(directory / "report.json")
        else:
            report_error = "Missing or unsafe config ID; artifacts not read."
        status, reason = suite_status(report, entry, report_error)
        counts[status] += 1
        previous_status = previous.get(identifier)
        change = result_change(status, previous_status)
        if change:
            changes[change] += 1
        anchor = pairing_anchor(identifier, index, used_anchors)
        failed = [str(check.get("name", "")) for check in phase_checks(report) if check.get("status") == "failed"]
        phase = next((str(check.get("name")) for check in phase_checks(report) if check.get("status") == "running"), "")
        failure = "; ".join(failed) or (reason if status in ("failed", "incomplete") else phase)
        search_text = " ".join(str(value) for value in pairing.values()) + " " + " ".join(
            str(value or "") for value in (
                entry.get("worker"), reason, report.get("error"), failure, previous_status, change,
            )
        )
        tile_report = report
        request_path = root / "inputs" / "requests" / f"{identifier}.json"
        if directory is not None and (
            not isinstance(report.get("request"), dict) or not report["request"]
        ) and local_file(root, request_path):
            request, _ = read_object(request_path)
            tile_report = {**report, "request": request, "_request_from_archive": True}
        tile_inputs.append((pairing, status, anchor, search_text, tile_report, entry))
        detail_inputs.append((
            pairing, entry, tile_report, status, reason, report_error, anchor, directory,
            previous_status, change,
        ))
        if status == "failed":
            failure_items.append(failure_item(pairing, status, anchor, tile_report, reason))
    terminal = bool(valid) and counts["passed"] + counts["failed"] == len(valid) and len(entries) == len(valid) and all(
        isinstance(entry, dict) and entry.get("status") in ("passed", "failed")
        and not entry.get("reserved") for entry in entries.values()
    )
    now = snapshot_time(ledger, terminal=terminal)
    tiles = [
        config_tile(pairing, status, anchor, search_text, config_runtime(report, entry, now), report)
        for pairing, status, anchor, search_text, report, entry in tile_inputs
    ]
    details = []
    for (
        pairing, entry, report, status, reason, report_error, anchor, directory,
        previous_status, change,
    ) in detail_inputs:
        details.append(config_details(root, pairing, entry, report, status, reason,
                                      report_error, anchor, directory, now,
                                      manifest.get("image_tag"), previous_status, change))
    peak, baseline, limit = (ledger.get(key) for key in ("peak_reserved_gpus", "baseline_gpus", "limit"))
    baseline = ledger.get("initial_baseline_gpus", baseline)
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (peak, limit)):
        if peak > limit:
            notices.append(f"Test GPU limit exceeded: peak reserved {peak} > {limit}.")
    observed_peak = ledger.get("peak_observed_account_gpus")
    budget_scope = ledger.get("gpu_budget_scope", manifest.get("gpu_budget_scope", "account"))
    if budget_scope != "sweep" and all(isinstance(value, (int, float)) and not isinstance(value, bool)
           for value in (observed_peak, limit)) and observed_peak > limit:
        notices.append(
            f"Account capacity reported {observed_peak} GPUs against the {limit}-GPU budget. "
            "The coordinator paused admissions on over-budget samples; capacity reporting "
            "lag or external allocations may differ from recorded sweep reservations."
        )
    summaries = "".join(
        f'<span><strong>{counts[status]}</strong>{badge(status)}</span>' for status in STATUSES
    )
    budget_scope_html = ""
    if budget_scope == "sweep":
        budget_scope_html = (
            '<details class="budget-audit"><summary>GPU accounting scope</summary>'
            '<p class="notice">The limit and test peak cover sweep-owned training and '
            'sampling reservations, retained through restore and verified cleanup. '
            'Admissions also require available capacity in the requested hardware pool. '
            'Unrelated jobs are included only in the separate observed account peak.</p></details>'
        )
    elif ledger.get("account_usage_samples"):
        budget_scope_html = (
            '<details class="budget-audit"><summary>GPU accounting scope</summary>'
            '<p class="notice">The peak covers retained sweep reservations plus sampled '
            'external account usage, including pending jobs across supported hardware pools. '
            'Capacity reporting lag and external submissions between samples are not '
            'controlled by the local coordinator.</p></details>'
        )
    options = "".join(f'<option value="{esc(model)}">{esc(model)}</option>' for model in models)
    status_options = "".join(f'<option value="{status}">{status.title()}</option>' for status in STATUSES)
    notice_html = "".join(f'<p class="notice">{esc(notice)}</p>' for notice in notices)
    comparison_html = ""
    if manifest.get("previous_run") is not None:
        previous_counts = Counter(previous.values())
        comparison_html = (
            '<section class="comparison" aria-label="Previous run comparison">'
            f'<p>Previous baseline <strong>{esc(manifest["previous_run"])}</strong></p>'
            f'<p>{len(previous)} configs &middot; {previous_counts["passed"]} passed'
            f' &middot; {previous_counts["failed"]} failed</p>'
            '<p class="comparison-counts">'
            + "".join(
                f'<span class="{label.lower()}">{label} <strong>{changes[label]}</strong></span>'
                for label in ("Fixed", "Regressed", "Unchanged")
            ) + '</p></section>'
        )
    stamp = pacific_time(now)
    live = bool(counts["queued"] or counts["running"])
    refresh_note = (
        "Auto-refreshes every 5 seconds while pairings are queued or running."
        if live else "Auto-refresh is off; all pairings have finished."
    )
    failure_rail = ""
    if failure_items:
        failure_rail = (
            '<aside class="failure-rail" aria-label="Failures">'
            f'<h2>Failures <span>{len(failure_items)}</span></h2>'
            f'<div class="failure-list">{"".join(failure_items)}</div></aside>'
        )
    main_class = ' class="has-failures"' if failure_items else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>Lifecycle suites - {esc(root.name)}</title>"
        + ('<meta http-equiv="refresh" content="5">' if live else '')
        + f"<style>{CSS}</style></head>"
        f"<body><main{main_class}>{failure_rail}<div class=\"report-body\">"
        '<div class="report-top"><header><div><h1>Lifecycle suites</h1>'
        f'<p>{len(valid)} configs &middot; {len(models)} models &middot; {esc(manifest.get("hardware", ""))}</p>'
        f'</div><p class="muted">Snapshot {esc(stamp)}<br>{esc(refresh_note)}</p></header>'
        f'<section class="summary" aria-label="Suite counts">{summaries}</section>'
        f'<div class="view-controls"><span id="visible-count" aria-live="polite">{len(valid)} / {len(valid)} configs</span></div>'
        f'</div>{notice_html}'
        f'<p class="image-tag">Requested matrix image tag <strong>{esc(manifest.get("image_tag") or "Deployment default")}</strong></p>'
        f'{comparison_html}'
        '<section class="metrics" aria-label="Test GPU metrics">'
        f'<span>Test GPU limit<strong>{esc(limit if limit is not None else "Not recorded")}</strong></span>'
        f'<span>Test GPU peak<strong>{esc(peak if peak is not None else "Not recorded")}</strong></span>'
        f'<span>Observed account GPU peak<strong>{esc(observed_peak if observed_peak is not None else "Not recorded")}</strong></span>'
        f'<span>Initial active GPUs<strong>{esc(baseline if baseline is not None else "Not recorded")}</strong></span>'
        f'</section>{budget_scope_html}'
        '<div class="controls" id="results"><label>Search<input id="search" type="search"></label>'
        f'<label>Status<select id="status-filter"><option value="">All statuses</option>{status_options}</select></label>'
        f'<label>Model<select id="model-filter"><option value="">All models</option>{options}</select></label>'
        '</div>'
        f'<section class="config-grid" aria-label="Configuration results">{"".join(tiles)}</section>'
        '<p id="no-results" hidden>No matching configs.</p>'
        f'{"".join(details)}'
        f"</div></main><script>{JS}</script></body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    if not root.is_dir():
        parser.error("run_root must be an existing run directory")
    output = root / "index.html"
    document = render(root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".index-", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(document)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
