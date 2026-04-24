import subprocess

try:
    with open("git_push_log.txt", "w") as f:
        print("Adding files...")
        subprocess.run(["git", "add", "."], check=False, stdout=f, stderr=subprocess.STDOUT)
        
        print("Committing...")
        subprocess.run(["git", "commit", "-am", "Finalize unified router, add history dashboard, include dataset"], check=False, stdout=f, stderr=subprocess.STDOUT)

        print("Pulling from remote...")
        subprocess.run(["git", "pull", "--rebase", "--autostash"], check=True, stdout=f, stderr=subprocess.STDOUT)
        
        print("Pushing to remote...")
        subprocess.run(["git", "push"], check=True, stdout=f, stderr=subprocess.STDOUT)
    print("Push succeeded")
except subprocess.CalledProcessError as e:
    print(f"Push failed with exit code {e.returncode}")
