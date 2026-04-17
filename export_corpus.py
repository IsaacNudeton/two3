#!/usr/bin/env python3
"""
export_corpus.py — Multi-track collaborator corpus exporter

Walks all Claude Code transcripts and per-project onetwo sessions,
produces layered training corpora for the duo model.

Outputs:
  corpus_dialogue.jsonl    — user<->assistant turns from *.jsonl transcripts
  corpus_methodology.txt   — onetwo reasoning chains per project
  manifest.json            — train/val split by project

Usage:
  python export_corpus.py --out ./corpus/
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from collections import defaultdict

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Hold-out projects for validation
VAL_PROJECTS = {
    "E--dev-learning-Organizing-Work",
    "E--dev-discord-lite",
}


def extract_text_only(content):
    """Extract ONLY natural-language text blocks. Skip tool_use/tool_result/thinking.
    Returns None if no real text content found."""
    if isinstance(content, str):
        s = content.strip()
        return s if len(s) >= 3 else None
    if isinstance(content, list):
        text_parts = []
        has_non_text = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                t = block.get("text", "").strip()
                if t:
                    text_parts.append(t)
            elif btype in ("tool_use", "tool_result", "thinking", "image"):
                has_non_text = True
        if text_parts:
            combined = "\n".join(text_parts).strip()
            return combined if len(combined) >= 3 else None
    return None


def export_dialogue(project_dir, out_samples):
    """Parse *.jsonl transcripts — only keep natural-language user<->assistant turns.
    Tool calls/results/thinking blocks are skipped (they're not dialogue)."""
    project_name = project_dir.name
    n_turns = 0
    for jsonl_file in sorted(project_dir.glob("*.jsonl")):
        session_id = jsonl_file.stem
        current_user = None
        with open(jsonl_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") not in ("user", "assistant"):
                    continue
                msg = d.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                text = extract_text_only(msg.get("content", ""))
                if text is None:
                    # Pure tool message — skip, don't break the pairing
                    continue
                if role == "user":
                    current_user = text
                elif role == "assistant" and current_user is not None:
                    out_samples.append({
                        "project": project_name,
                        "session": session_id,
                        "track": "dialogue",
                        "user": current_user,
                        "assistant": text,
                    })
                    n_turns += 1
                    current_user = None
    return n_turns


def export_onetwo(project_path, project_name):
    """Run `onetwo status` in a project dir to dump its reasoning chain."""
    xyzt_dir = project_path / ".xyzt"
    if not (xyzt_dir / "onetwo_session.bin").exists():
        return None
    try:
        result = subprocess.run(
            ["onetwo", "status"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0 and "ONETWO-STATUS" in result.stdout:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def find_project_workdirs():
    """Find all dev dirs with .xyzt subdirs."""
    roots = [Path("E:/dev"), Path("C:/dev"), Path.home() / "dev"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for xyzt_dir in root.rglob(".xyzt"):
                if xyzt_dir.is_dir():
                    found.append(xyzt_dir.parent)
        except (PermissionError, OSError):
            continue
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./corpus", help="output directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Scanning Claude transcripts in {CLAUDE_PROJECTS}")
    dialogue_samples = []
    project_counts = defaultdict(int)
    if CLAUDE_PROJECTS.exists():
        for project_dir in sorted(CLAUDE_PROJECTS.iterdir()):
            if not project_dir.is_dir():
                continue
            n = export_dialogue(project_dir, dialogue_samples)
            if n > 0:
                project_counts[project_dir.name] = n
                print(f"  {project_dir.name}: {n} turns")

    print(f"\n[2/3] Scanning onetwo sessions in dev project dirs")
    methodology_samples = []
    workdirs = find_project_workdirs()
    for workdir in sorted(workdirs):
        status = export_onetwo(workdir, workdir.name)
        if status:
            methodology_samples.append({
                "project": workdir.name,
                "path": str(workdir),
                "track": "methodology",
                "content": status,
            })
            print(f"  {workdir.name}: captured")

    print(f"\n[3/3] Writing outputs to {out_dir}/")

    total_dialogue = len(dialogue_samples)
    total_methodology = len(methodology_samples)
    print(f"  dialogue samples: {total_dialogue}")
    print(f"  methodology samples: {total_methodology}")

    if args.dry_run:
        print("\n(dry run — not writing files)")
        return

    # Write dialogue corpus
    dialogue_path = out_dir / "corpus_dialogue.jsonl"
    with open(dialogue_path, "w", encoding="utf-8") as f:
        for s in dialogue_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    dialogue_bytes = dialogue_path.stat().st_size
    print(f"  wrote {dialogue_path} ({dialogue_bytes/1024/1024:.1f} MB)")

    # Write methodology corpus
    methodology_path = out_dir / "corpus_methodology.jsonl"
    with open(methodology_path, "w", encoding="utf-8") as f:
        for s in methodology_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    methodology_bytes = methodology_path.stat().st_size
    print(f"  wrote {methodology_path} ({methodology_bytes/1024/1024:.1f} MB)")

    # Write manifest
    train_projects = [p for p in project_counts if p not in VAL_PROJECTS]
    val_projects = [p for p in project_counts if p in VAL_PROJECTS]
    manifest = {
        "dialogue": {
            "path": "corpus_dialogue.jsonl",
            "total_samples": total_dialogue,
            "total_bytes": dialogue_bytes,
            "projects": dict(project_counts),
            "train_projects": train_projects,
            "val_projects": val_projects,
        },
        "methodology": {
            "path": "corpus_methodology.jsonl",
            "total_samples": total_methodology,
            "total_bytes": methodology_bytes,
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {manifest_path}")

    print(f"\nDone. Train projects: {len(train_projects)}  Val projects: {len(val_projects)}")


if __name__ == "__main__":
    main()
