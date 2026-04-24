import subprocess

try:
    print("Aborting any existing rebase...")
    subprocess.run(["git", "rebase", "--abort"], capture_output=True)
    
    print("Committing unstaged changes...")
    subprocess.run(["git", "commit", "-am", "Auto-commit before push"], capture_output=True)

    print("Pushing to remote (force)...")
    result_push = subprocess.run(
        ["git", "push", "--force"], 
        capture_output=True, 
        text=True,
        check=True
    )
    print("Push stdout:", result_push.stdout)
    print("Push succeeded")
    
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
