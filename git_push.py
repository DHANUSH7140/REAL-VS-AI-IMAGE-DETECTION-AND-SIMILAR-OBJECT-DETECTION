import subprocess

try:
    print("Committing unstaged changes...")
    subprocess.run(["git", "commit", "-am", "Auto-commit before pull"], capture_output=True)

    print("Pulling from remote...")
    result_pull = subprocess.run(
        ["git", "pull", "--rebase"], 
        capture_output=True, 
        text=True,
        check=True
    )
    print("Pull stdout:", result_pull.stdout)
    
    print("Pushing to remote...")
    result_push = subprocess.run(
        ["git", "push"], 
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
