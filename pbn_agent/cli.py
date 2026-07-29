from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, load_settings
from .models import StageName, WorkflowSpec
from .workflow import (
    align_current_stage_a,
    approve_stage,
    finalize_run,
    generate_stage,
    initialize_run,
    load_state,
    plan_run,
    rebuild_current_stage_c,
    stage_request,
)


def _stage(value: str) -> StageName:
    normalized = value.upper()
    if normalized not in {"A", "B", "C", "D", "E"}:
        raise argparse.ArgumentTypeError("stage must be A, B, C, D, or E")
    return normalized  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paint-by-numbers single-agent MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a new workflow run")
    init.add_argument("--input", type=Path, required=True)
    init.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    init.add_argument("--run-id")

    plan = subparsers.add_parser("plan", help="Create the typed A-E plan")
    plan.add_argument("--run", type=Path, required=True)
    plan.add_argument("--offline", action="store_true", help="Use the checked-in baseline prompts")

    generate = subparsers.add_parser("generate", help="Generate or revise one image stage")
    generate.add_argument("--run", type=Path, required=True)
    generate.add_argument("--stage", type=_stage, required=True)
    generate.add_argument("--feedback")
    generate.add_argument("--prompt-only", action="store_true")

    align_a = subparsers.add_parser("align-a", help="Align the current chroma-key Stage A to source geometry")
    align_a.add_argument("--run", type=Path, required=True)

    rebuild_c = subparsers.add_parser("rebuild-c", help="Rebuild current C without consuming a revision")
    rebuild_c.add_argument("--run", type=Path, required=True)

    approve = subparsers.add_parser("approve", help="Approve stage C or E")
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--stage", type=_stage, required=True)

    finalize = subparsers.add_parser("finalize", help="Generate 36-color production files after E approval")
    finalize.add_argument("--run", type=Path, required=True)
    finalize.add_argument("--protected-mask", type=Path)

    status = subparsers.add_parser("status", help="Show run state without secrets")
    status.add_argument("--run", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = load_settings()
    if args.command == "init":
        run = initialize_run(args.input, args.output_root, WorkflowSpec(), args.run_id)
        print(run)
    elif args.command == "plan":
        print(plan_run(args.run.resolve(), settings, offline=args.offline))
    elif args.command == "generate":
        run = args.run.resolve()
        if args.prompt_only:
            references, prompt, output, attempt = stage_request(run, args.stage, args.feedback)
            print(json.dumps({
                "stage": args.stage,
                "attempt": attempt,
                "references": [str(path) for path in references],
                "output": str(output),
                "prompt": prompt,
            }, ensure_ascii=False, indent=2))
        else:
            print(generate_stage(run, args.stage, settings, args.feedback))
    elif args.command == "align-a":
        print(align_current_stage_a(args.run.resolve()))
    elif args.command == "rebuild-c":
        print(rebuild_current_stage_c(args.run.resolve()))
    elif args.command == "approve":
        approve_stage(args.run.resolve(), args.stage)
        print(f"approved stage {args.stage}")
    elif args.command == "finalize":
        result = finalize_run(args.run.resolve(), args.protected_mask)
        print(result.numbered_line_art)
    elif args.command == "status":
        print(load_state(args.run.resolve()).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
