import os
import subprocess
import urllib.request
import time
import sys

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
BFG_JAR = "bfg.jar"

def download_bfg():
    if not os.path.exists(BFG_JAR):
        print("📦 Downloading BFG...")
        url = "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar"
        urllib.request.urlretrieve(url, BFG_JAR)
        print("✅ BFG downloaded.")

def find_large_files():
    print("🔍 Scanning for files > 100MB in Git history...")
    cmd = (
        "git rev-list --objects --all | "
        "git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize)' | "
        f"grep '^blob' | awk '$3 >= {MAX_FILE_SIZE}'"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()

def run_bfg_cleanup():
    print("🧹 Cleaning Git history using BFG...")
    subprocess.run("git clone --mirror . temp_repo", shell=True, check=True)
    subprocess.run(f"java -jar {BFG_JAR} --delete-folders .venv --delete-files '*torch_cpu.dll,*dnnl.lib' temp_repo", shell=True, check=True)
    subprocess.run("cd temp_repo && git reflog expire --expire=now --all && git gc --prune=now --aggressive", shell=True, check=True)
    subprocess.run("cd temp_repo && git push --force", shell=True, check=True)

def try_push():
    print("🚀 Attempting push...")
    result = subprocess.run("git push -u origin main", shell=True)
    return result.returncode == 0

def main():
    download_bfg()
    attempt = 1

    while True:
        print(f"\n🔁 Attempt #{attempt}")
        large_files = find_large_files()

        if large_files:
            print(f"⚠️ Found {len(large_files)} large file(s). Running BFG cleanup.")
            run_bfg_cleanup()
        else:
            print("✅ No large files detected.")

        if try_push():
            print("✅ Push succeeded!")
            break
        else:
            print("❌ Push failed. Retrying in 5s...")
            time.sleep(5)
            attempt += 1
            if attempt > 3:
                print("🛑 Failed after 3 attempts. Exiting.")
                sys.exit(1)

if __name__ == "__main__":
    main() 