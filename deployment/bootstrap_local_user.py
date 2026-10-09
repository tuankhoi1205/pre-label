"""Create a local CVAT admin without putting its password in logs/arguments."""
import argparse
import json
import secrets
import subprocess
from pathlib import Path


def bootstrap(docker):
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        password = secrets.token_urlsafe(24)
        with env_path.open("x", encoding="utf-8") as stream:
            stream.write(
                f"NUSCENES_ROOT={root.as_posix()}/data/nuscenes\n"
                "CVAT_URL=http://localhost:8080\nCVAT_AUTH_SCHEME=Token\n"
                f"CVAT_USERNAME=admin\nCVAT_PASSWORD={password}\nCVAT_ORG=\n"
            )
    values = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            values[key.strip()] = value.strip().strip("\"'")
    if not values.get("CVAT_USERNAME") or not values.get("CVAT_PASSWORD"):
        raise ValueError("Set CVAT_USERNAME/CVAT_PASSWORD in .env first")
    code = '''import json, os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cvat.settings.production")
import django
django.setup()
from django.contrib.auth import get_user_model
p = json.load(sys.stdin)
User = get_user_model()
user = User.objects.filter(username=p["username"]).first()
if user is None:
    User.objects.create_superuser(p["username"], "admin@localhost", p["password"])
    print("Local CVAT admin created")
elif user.check_password(p["password"]):
    print("Local CVAT credentials verified")
else:
    raise SystemExit("Existing user has different credentials; its password was not changed")
'''
    subprocess.run([docker, "exec", "-i", "cvat_server", "python3", "-c", code],
        input=json.dumps({"username": values["CVAT_USERNAME"], "password": values["CVAT_PASSWORD"]}),
        text=True, check=True)
    print(f"Credentials saved in {env_path} (password omitted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", required=True)
    bootstrap(parser.parse_args().docker)
