import json
import os
import sys

import requests

WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")


def send_slack_alert(message: str) -> int:
    if not WEBHOOK_URL:
        print("WARNING: SLACK_WEBHOOK_URL is not set; skipping Slack alert.", file=sys.stderr)
        return 0

    payload = {"text": message}
    response = requests.post(
        WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    return response.status_code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slack.py <message>", file=sys.stderr)
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    # Prefer stdin for multi-line payloads when "-" is passed.
    if message == "-":
        message = sys.stdin.read()
    status = send_slack_alert(message)
    if status and status >= 400:
        print(f"Slack webhook returned HTTP {status}", file=sys.stderr)
        sys.exit(1)
