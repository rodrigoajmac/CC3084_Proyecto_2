"""Manual PKCE OAuth flow for Kaggle's MCP server.

Works around Kaggle's registration endpoint requiring an explicit
token_endpoint_auth_method:"none" while its AS metadata omits
token_endpoint_auth_methods_supported (so clients default to
client_secret_basic per RFC 8414 and fail with no secret).
"""
import base64, hashlib, http.server, json, os, secrets, socket
import urllib.parse, urllib.request, urllib.error

REG = "https://www.kaggle.com/api/v1/oauth2/register"
AUTH = "https://www.kaggle.com/api/v1/oauth2/authorize"
TOKEN = "https://www.kaggle.com/api/v1/oauth2/token"
RESOURCE = "https://www.kaggle.com/mcp"
SCOPE = "resources.admin:*"
PORT = 8765
REDIRECT = "http://localhost:%d/callback" % PORT
OUT = os.path.join(os.path.expanduser("~"), ".kaggle-mcp-token.json")


def post_json(url, obj):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def post_form(url, fields):
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


# 1. Register a public client. The explicit "none" is the whole point.
name = "claude-code-mcp-" + secrets.token_hex(5)
reg = post_json(REG, {
    "client_name": name,
    "redirect_uris": [REDIRECT],
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"],
    "token_endpoint_auth_method": "none",
})
client_id = reg["client_id"]
print("registered client_id: %s" % client_id, flush=True)

# 2. PKCE + state.
verifier = b64url(secrets.token_bytes(32))
challenge = b64url(hashlib.sha256(verifier.encode()).digest())
state = secrets.token_urlsafe(16)

url = AUTH + "?" + urllib.parse.urlencode({
    "response_type": "code",
    "client_id": client_id,
    "code_challenge": challenge,
    "code_challenge_method": "S256",
    "redirect_uri": REDIRECT,
    "state": state,
    "scope": SCOPE,
    "resource": RESOURCE,
})
print("\n=== OPEN THIS URL IN YOUR BROWSER ===\n%s\n" % url, flush=True)

# 3. One-shot listener for the redirect.
captured = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        captured.update({k: v[0] for k, v in q.items()})
        ok = "code" in captured and captured.get("state") == state
        body = ("<h2>%s</h2><p>You can close this tab and return to Claude Code.</p>"
                % ("Authorized." if ok else "Authorization failed.")).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
srv.timeout = 300
print("waiting for redirect on %s (5 min timeout)..." % REDIRECT, flush=True)
srv.handle_request()

if "code" not in captured:
    raise SystemExit("no authorization code received: %s" % captured)
if captured.get("state") != state:
    raise SystemExit("state mismatch - aborting")

# 4. Exchange. Public client: client_id in the body, no secret, no Basic header.
try:
    tok = post_form(TOKEN, {
        "grant_type": "authorization_code",
        "code": captured["code"],
        "redirect_uri": REDIRECT,
        "client_id": client_id,
        "code_verifier": verifier,
        "resource": RESOURCE,
    })
except urllib.error.HTTPError as e:
    raise SystemExit("token exchange failed %s: %s" % (e.code, e.read().decode()[:500]))

tok["client_id"] = client_id
tok["redirect_uri"] = REDIRECT
with open(OUT, "w") as f:
    json.dump(tok, f, indent=2)
try:
    os.chmod(OUT, 0o600)
except OSError:
    pass

print("\nSUCCESS - token saved to %s" % OUT, flush=True)
print("  token_type:    %s" % tok.get("token_type"), flush=True)
print("  expires_in:    %s" % tok.get("expires_in"), flush=True)
print("  refresh_token: %s" % ("yes" if tok.get("refresh_token") else "no"), flush=True)
print("  scope:         %s" % tok.get("scope"), flush=True)
