import subprocess

print("Configuring git...")
subprocess.run(["git", "config", "core.safecrlf", "false"], check=False)

print("Adding files...")
subprocess.run(["git", "add", "."], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Committing...")
try:
    subprocess.run(["git", "commit", "-m", "Finalize unified router, add history dashboard, include dataset"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    print("Nothing to commit")

print("Pushing...")
subprocess.run(["git", "push"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Done!")
