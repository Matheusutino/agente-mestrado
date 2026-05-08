from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    request = Request(
        OPENROUTER_CREDITS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"OpenRouter request failed with HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"OpenRouter request failed: {exc}", file=sys.stderr)
        return 1

    data = payload.get("data", {})
    total_credits = float(data.get("total_credits", 0.0))
    total_usage = float(data.get("total_usage", 0.0))
    balance = total_credits - total_usage

    print(f"total_credits: {total_credits:.4f}")
    print(f"total_usage:   {total_usage:.4f}")
    print(f"balance:       {balance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
