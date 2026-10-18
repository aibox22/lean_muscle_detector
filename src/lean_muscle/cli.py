"""Score one local photo. Prints the JSON and writes the debug image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_bytes
from .errors import GatewayError, MissingKey, Refuse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给一张正面照打薄肌分")
    parser.add_argument("image", type=Path, help="JPEG、PNG 或 WebP")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="调试图目录")
    parser.add_argument("--no-summary", action="store_true", help="不调用总结模型")
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"找不到 {args.image}", file=sys.stderr)
        return 2
    try:
        result = analyze_bytes(args.image.read_bytes(), summarize=not args.no_summary)
    except Refuse as exc:
        print(json.dumps({"ok": False, "reason": exc.reason, "detail": exc.detail}, ensure_ascii=False))
        return 2
    except MissingKey as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except GatewayError as exc:
        print(json.dumps({"ok": False, "reason": "gateway_error", "detail": str(exc)}, ensure_ascii=False))
        return 3

    args.output.mkdir(parents=True, exist_ok=True)
    png_path = args.output / f"{args.image.stem}.png"
    png_path.write_bytes(result.debug_png)
    payload = dict(result.body)
    payload["ok"] = True
    payload["calibrated"] = False
    payload["debug_image"] = str(png_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
