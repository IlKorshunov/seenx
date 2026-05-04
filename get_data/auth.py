from __future__ import annotations

from pathlib import Path

import requests
from google_auth_oauthlib.flow import InstalledAppFlow


_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = _ROOT / "credentials.json"
TOKEN_PATH = _ROOT / "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

_DNS_HINT = """
DNS не резолвит хосты Google (ошибка при обмене code → token на oauth2.googleapis.com).

Исправь системный DNS (например 8.8.8.8 / 1.1.1.1 в NetworkManager) или временно в /etc/hosts
(уточни IP: host oauth2.googleapis.com 8.8.8.8), например:

  sudo sh -c 'echo "64.233.161.95 oauth2.googleapis.com" >> /etc/hosts'

После успешного получения токена строку можно удалить, когда починишь DNS.
"""


def main() -> None:
    if not CREDENTIALS_PATH.is_file():
        raise FileNotFoundError(f"Нет файла: {CREDENTIALS_PATH}")

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
    try:
        creds = flow.run_local_server(port=0)
    except (requests.exceptions.ConnectionError, OSError) as e:
        err = str(e).lower()
        if "name resolution" in err or "failed to resolve" in err or "errno -3" in err or "gaierror" in err:
            print(_DNS_HINT)
        raise

    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"Готово: {TOKEN_PATH}")


if __name__ == "__main__":
    main()
