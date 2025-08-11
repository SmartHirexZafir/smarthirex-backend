# backend/services/code_runner.py
"""
Secure-ish code execution helper for coding questions.

Key features
------------
- Supports Python, Node.js, Java, and C++ out of the box (easily extensible).
- Per-test timeouts, process group kill, and (POSIX) CPU/memory limits.
- Runs each submission in an isolated temp directory.
- Optional simple output matchers: exact / contains / regex.
- Pure-Python module with a clean API: run_submission(submission) -> dict.
- Can also be used as a CLI: `python code_runner.py payload.json`

Security & Notes
----------------
This is a *constrained* local runner. It is not a full sandbox:
- It uses `resource` limits (Unix) and tight timeouts, but it is not a VM.
- For hostile workloads, prefer Docker/Firecracker and drop caps / seccomp.
- Ensure your API layer sanitizes inputs and applies auth/quotas.

Submission schema (example)
---------------------------
{
  "language": "python",                # one of: python, node, java, cpp
  "source": "print(input())",          # code as string
  "tests": [
    {
      "name": "echo 1",
      "input": "hello\n",
      "args": [],                      # optional argv
      "expected_stdout": "hello",
      "match": "exact"                 # exact | contains | regex
    }
  ],
  "time_limit_sec": 2,                 # default per test
  "memory_limit_mb": 128               # approximate RSS limit
}

Return shape
------------
{
  "ok": true/false,
  "language": "python",
  "compile": {"ok": true, "stdout": "", "stderr": "", "time_sec": 0.021},
  "tests": [
    {
      "name": "echo 1",
      "ok": true,
      "timeout": false,
      "stdout": "hello\n",
      "stderr": "",
      "time_sec": 0.005,
      "match_result": {"mode":"exact","passed":true,"expected":"hello","detail":""}
    },
    ...
  ],
  "summary": {"passed": 1, "total": 1}
}
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----- POSIX resource limits (best-effort) -----
try:
    import resource  # type: ignore
    _HAS_RESOURCE = True
except Exception:  # pragma: no cover
    _HAS_RESOURCE = False


# ---------------- Language configs ----------------

@dataclass
class LanguageSpec:
    ext: str
    # compile: (cmd, output_binary_path) or None for interpreted
    compile_cmd: Optional[str] = None
    run_cmd: Optional[str] = None  # use {binary} placeholder if binary exists
    binary: Optional[str] = None   # name of the compiled artifact if any


LANGS: Dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        ext="py",
        compile_cmd=None,
        run_cmd="{python} main.py",
    ),
    "node": LanguageSpec(
        ext="js",
        compile_cmd=None,
        run_cmd="{node} main.js",
    ),
    "java": LanguageSpec(
        ext="java",
        compile_cmd="javac Main.java",
        run_cmd="java Main",
        binary=None,  # .class files are generated
    ),
    "cpp": LanguageSpec(
        ext="cpp",
        compile_cmd="g++ -std=c++17 -O2 -pipe -static -s main.cpp -o main",
        run_cmd="./main",
        binary="main",
    ),
}


# ---------------- Match helpers ----------------

@dataclass
class MatchResult:
    mode: str
    passed: bool
    expected: Optional[str] = None
    detail: str = ""


def match_output(mode: str, expected: Optional[str], actual: str) -> MatchResult:
    mode = (mode or "exact").lower()
    if expected is None:
        return MatchResult(mode=mode, passed=True, expected=None, detail="no expectation")

    if mode == "exact":
        return MatchResult(mode="exact", passed=(actual.strip() == expected.strip()),
                           expected=expected, detail="strict string equality")
    if mode == "contains":
        return MatchResult(mode="contains", passed=(expected in actual),
                           expected=expected, detail="substring containment")
    if mode == "regex":
        try:
            ok = re.search(expected, actual, flags=re.MULTILINE) is not None
            return MatchResult(mode="regex", passed=ok, expected=expected, detail="regex search")
        except re.error as e:
            return MatchResult(mode="regex", passed=False, expected=expected, detail=f"bad regex: {e}")
    # default
    return MatchResult(mode=mode, passed=False, expected=expected, detail=f"unknown mode '{mode}'")


# ---------------- Core runner ----------------

@dataclass
class RunLimits:
    time_limit_sec: int = 2
    memory_limit_mb: int = 128


def _set_limits(limits: RunLimits):
    """Child-only: apply resource limits for safety (POSIX)."""
    if not _HAS_RESOURCE:
        return
    # CPU time (seconds). Add 1s grace so Python's timeout still applies.
    cpu = max(1, limits.time_limit_sec)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    # Address space / virtual memory
    bytes_limit = max(32, limits.memory_limit_mb) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    except Exception:
        # Some platforms disallow setting RLIMIT_AS; ignore.
        pass
    # Open files
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _popen(cmd: str, cwd: Path, limits: RunLimits, env: Dict[str, str]) -> subprocess.Popen:
    preexec_fn = _set_limits if os.name == "posix" else None
    # Start a new process group so we can kill children on timeout.
    return subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec_fn,  # type: ignore[arg-type]
        start_new_session=True
    )


def _terminate_tree(proc: subprocess.Popen):
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    time_sec: float
    timeout: bool = False


def _run_cmd(cmd: str, cwd: Path, limits: RunLimits, input_data: Optional[str], env: Dict[str, str]) -> ExecResult:
    start = time.time()
    proc = _popen(cmd, cwd, limits, env)
    try:
        out, err = proc.communicate(input=input_data, timeout=limits.time_limit_sec)
        ok = proc.returncode == 0
        return ExecResult(ok=ok, stdout=out, stderr=err, time_sec=time.time() - start)
    except subprocess.TimeoutExpired:
        _terminate_tree(proc)
        return ExecResult(ok=False, stdout="", stderr="TIMEOUT", time_sec=time.time() - start, timeout=True)


# ---------------- Public API ----------------

@dataclass
class TestCase:
    name: str
    input: str = ""
    args: List[str] = field(default_factory=list)
    expected_stdout: Optional[str] = None
    match: str = "exact"
    timeout_sec: Optional[int] = None  # overrides global


def _write_source(base: Path, lang: LanguageSpec, source: str) -> Path:
    fname = "main." + lang.ext
    path = base / fname
    # Normalize line endings, strip BOM, etc.
    cleaned = source.replace("\r\n", "\n").lstrip("\ufeff")
    path.write_text(cleaned, encoding="utf-8")
    # Java must have class Main
    if lang is LANGS["java"] and "class Main" not in cleaned:
        raise ValueError("Java submissions must contain a public class named 'Main'.")
    return path


def _env_paths() -> Dict[str, str]:
    env = os.environ.copy()
    # Allow callers to inject specific runtimes via env
    env.setdefault("PYTHON_BIN", sys.executable)
    env.setdefault("NODE_BIN", "node")
    return env


def _build_commands(lang_key: str, workdir: Path) -> Tuple[LanguageSpec, Optional[str], str]:
    if lang_key not in LANGS:
        raise ValueError(f"Unsupported language: {lang_key}")
    spec = LANGS[lang_key]
    python_bin = _env_paths()["PYTHON_BIN"]
    node_bin = _env_paths()["NODE_BIN"]

    run_cmd = spec.run_cmd or ""
    run_cmd = run_cmd.replace("{python}", shlex.quote(python_bin)).replace("{node}", shlex.quote(node_bin))
    compile_cmd = spec.compile_cmd
    if compile_cmd:
        compile_cmd = compile_cmd.format().strip()
    return spec, compile_cmd, run_cmd


def run_submission(payload: Dict) -> Dict:
    """
    Compile and/or run code against test cases.

    payload keys:
      - language: str
      - source: str
      - tests: List[dict] (optional)
      - time_limit_sec: int (optional)
      - memory_limit_mb: int (optional)

    Returns a dict as described in the module docstring.
    """
    language = str(payload.get("language", "")).lower().strip()
    source = payload.get("source", "")
    raw_tests = payload.get("tests", []) or []
    limits = RunLimits(
        time_limit_sec=int(payload.get("time_limit_sec", 2)),
        memory_limit_mb=int(payload.get("memory_limit_mb", 128)),
    )

    tests: List[TestCase] = []
    for t in raw_tests:
        tests.append(TestCase(
            name=str(t.get("name", "test")),
            input=str(t.get("input", "")),
            args=list(t.get("args", []) or []),
            expected_stdout=t.get("expected_stdout"),
            match=str(t.get("match", "exact")),
            timeout_sec=t.get("timeout"),
        ))

    result: Dict = {
        "ok": False,
        "language": language,
        "compile": {},
        "tests": [],
        "summary": {"passed": 0, "total": len(tests)}
    }

    with tempfile.TemporaryDirectory(prefix="runner_") as tmp:
        workdir = Path(tmp)

        # 1) Write source
        try:
            spec, compile_cmd, run_cmd = _build_commands(language, workdir)
            _write_source(workdir, spec, source)
        except Exception as e:
            result["compile"] = {
                "ok": False,
                "stdout": "",
                "stderr": f"Source/write error: {e}",
                "time_sec": 0.0
            }
            return result

        env = _env_paths()

        # 2) Compile if needed
        if compile_cmd:
            comp = _run_cmd(compile_cmd, workdir, limits, input_data=None, env=env)
            result["compile"] = comp.__dict__
            if not comp.ok:
                return result
        else:
            result["compile"] = {"ok": True, "stdout": "", "stderr": "", "time_sec": 0.0}

        # 3) Run tests
        passed = 0
        for t in tests:
            t_limits = RunLimits(
                time_limit_sec=int(t.timeout_sec or limits.time_limit_sec),
                memory_limit_mb=limits.memory_limit_mb
            )
            # Build command with args
            cmd = run_cmd
            if t.args:
                cmd = f"{cmd} {' '.join(shlex.quote(str(a)) for a in t.args)}"

            ex = _run_cmd(cmd, workdir, t_limits, input_data=t.input, env=env)
            m = match_output(t.match, t.expected_stdout, ex.stdout)

            case_out = {
                "name": t.name,
                "ok": ex.ok and m.passed,
                "timeout": ex.timeout,
                "stdout": ex.stdout,
                "stderr": ex.stderr,
                "time_sec": ex.time_sec,
                "match_result": {
                    "mode": m.mode,
                    "passed": m.passed,
                    "expected": m.expected,
                    "detail": m.detail
                }
            }
            if case_out["ok"]:
                passed += 1
            result["tests"].append(case_out)

        result["summary"]["passed"] = passed
        result["ok"] = (passed == len(tests)) and result["compile"].get("ok", False)
        return result


# ---------------- CLI ----------------

def _read_json(path: Optional[str]) -> Dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    # stdin
    data = sys.stdin.read()
    return json.loads(data)


def _demo_payload(lang: str = "python") -> Dict:
    return {
        "language": lang,
        "source": textwrap.dedent("""
            # echo program
            import sys
            data = sys.stdin.read()
            print(data.strip())
        """),
        "tests": [
            {"name": "t1", "input": "hello", "expected_stdout": "hello", "match": "exact"}
        ],
        "time_limit_sec": 2,
        "memory_limit_mb": 128
    }


if __name__ == "__main__":
    """
    Example:
      python code_runner.py payload.json
      echo '{"language":"python","source":"print(1)","tests":[{"name":"t","expected_stdout":"1"}]}' | python code_runner.py
      python code_runner.py --demo java
    """
    if len(sys.argv) >= 2 and sys.argv[1] == "--demo":
        lang = sys.argv[2] if len(sys.argv) >= 3 else "python"
        payload = _demo_payload(lang)
        print(json.dumps(run_submission(payload), indent=2))
        sys.exit(0)

    payload_path = sys.argv[1] if len(sys.argv) >= 2 else None
    try:
        payload = _read_json(payload_path)
        out = run_submission(payload)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
