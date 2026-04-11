import instaloader
from requests.cookies import RequestsCookieJar
from dotenv import load_dotenv
import os

load_dotenv()

SESSION_ID = os.getenv("SESSION_ID")
CSRF_TOKEN = os.getenv("CSRF_TOKEN")
DS_USER_ID = os.getenv("DS_USER_ID")
MID = os.getenv("MID")

if not all([SESSION_ID, CSRF_TOKEN, DS_USER_ID, MID]):
    raise ValueError("Variabili mancanti nel file .env. Controlla SESSION_ID, CSRF_TOKEN, DS_USER_ID, MID.")

cookie_string = (
    f"sessionid={SESSION_ID}; "
    f"csrftoken={CSRF_TOKEN}; "
    f"ds_user_id={DS_USER_ID}; "
    f"mid={MID}"
)

cookies = {}
for item in cookie_string.split("; "):
    if "=" in item:
        name, value = item.split("=", 1)
        cookies[name] = value

jar = RequestsCookieJar()
for name, value in cookies.items():
    jar.set(name, value, domain=".instagram.com", path="/")

L = instaloader.Instaloader()
L.context._session.cookies.update(jar)

username = L.test_login()
print("Logged in as:", username)

profile = instaloader.Profile.from_username(L.context, username)
print("Profile id:", profile.userid)

L.context.username = username
L.save_session_to_file("session.txt")
print("Session saved.")