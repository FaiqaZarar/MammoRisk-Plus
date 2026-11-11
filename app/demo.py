"""Local demo launcher: python -m app.demo

Starts the platform on http://127.0.0.1:5055 with a throwaway hospital login created for this run only.
The password is random, printed once in the console, and never stored. For real deployments set
MAMMORISK_USERS / MAMMORISK_SECRET_KEY instead (see Dockerfile). Do not expose this to a network.
"""
import secrets

from werkzeug.security import generate_password_hash

from app import create_app

if __name__ == "__main__":
    password = secrets.token_urlsafe(9)
    # DEMO_MODE: result pages show clearly stamped ILLUSTRATIVE sample output while no model is installed.
    app = create_app({"USERS": {"demo": generate_password_hash(password)}, "DEMO_MODE": True})
    app.config["TEMPLATES_AUTO_RELOAD"] = True   # edit a template, then just refresh the page
    print(f"\nMammoRisk+ demo -> http://127.0.0.1:5055   login: demo / {password}\n"
          "Demo mode: result pages show sample output stamped ILLUSTRATIVE (not computed from your files).\n")
    app.run(host="127.0.0.1", port=5055)
