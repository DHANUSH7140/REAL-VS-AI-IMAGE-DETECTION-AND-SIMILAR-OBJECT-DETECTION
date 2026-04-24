import subprocess
import sys

def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-bcrypt", "--quiet"])

if __name__ == "__main__":
    install()
