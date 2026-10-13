"""Refresh / print the Kaggle MCP access token.

Kaggle issues 3-hour access tokens. Run this when the token expires:

    python scripts/kaggle_mcp_token.py            # refresh if stale, print status
    python scripts/kaggle_mcp_token.py --print    # print the raw token
    python scripts/kaggle_mcp_token.py --setx     # persist to the KAGGLE_MCP_TOKEN user env var

Re-run the full authorization (scripts/kaggle_mcp_login.py) only if the
refresh token itself is rejected.
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

TOKEN_URL = "https://www.kaggle.com/api/v1/oauth2/token"
RESOURCE = "https://www.kaggle.com/mcp"
STORE = os.path.join(os.path.expanduser("~"), ".kaggle-mcp-token.json")


def load():
    if not os.path.exists(STORE):
        sys.exit("no token store at %s - run scripts/kaggle_mcp_login.py first" % STORE)
    with open(STORE) as f:
        return json.load(f)


def save(tok):
    with open(STORE, "w") as f:
        json.dump(tok, f, indent=2)
    try:
        os.chmod(STORE, 0o600)
    except OSError:
        pass


def refresh(tok):
    if not tok.get("refresh_token"):
        sys.exit("no refresh_token stored - re-run scripts/kaggle_mcp_login.py")
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": tok["refresh_token"],
        "client_id": tok["client_id"],
        "resource": RESOURCE,
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        new = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        sys.exit("refresh failed %s: %s\nre-run scripts/kaggle_mcp_login.py"
                 % (e.code, e.read().decode()[:300]))
    # Kaggle may or may not rotate the refresh token; keep the old one if absent.
    new.setdefault("refresh_token", tok["refresh_token"])
    new["client_id"] = tok["client_id"]
    new["redirect_uri"] = tok.get("redirect_uri")
    new["obtained_at"] = int(time.time())
    save(new)
    return new


def main():
    tok = load()
    age = int(time.time()) - tok.get("obtained_at", 0)
    stale = "obtained_at" not in tok or age > tok.get("expires_in", 10800) - 300

    if "--print" in sys.argv:
        if stale:
            tok = refresh(tok)
        sys.stdout.write(tok["access_token"])
        return

    if stale:
        tok = refresh(tok)
        print("refreshed - new token valid for %ss" % tok.get("expires_in"))
    else:
        left = tok.get("expires_in", 10800) - age
        print("token still valid for ~%dm" % (left // 60))

    if "--setx" in sys.argv:
        if os.name != "nt":
            sys.exit("--setx is Windows-only; use: export KAGGLE_MCP_TOKEN=$(python scripts/kaggle_mcp_token.py --print)")
        os.system('setx KAGGLE_MCP_TOKEN "%s" >nul' % tok["access_token"])
        print("KAGGLE_MCP_TOKEN set for your user account.")
        print("Restart Claude Code (and any open terminals) to pick it up.")


if __name__ == "__main__":
    main()
