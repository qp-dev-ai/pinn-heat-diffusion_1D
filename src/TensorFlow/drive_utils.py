import os
import string
import subprocess


def find_free_drive_letter():
    used = set()

    # disques existants
    for letter in string.ascii_uppercase:
        if os.path.exists(f"{letter}:\\"):
            used.add(letter)

    # disques subst
    result = subprocess.run("subst", capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if ":" in line:
            used.add(line[0])

    preferred = list("TUVWXYZ") + list(string.ascii_uppercase)

    for letter in preferred:
        if letter not in used:
            return f"{letter}:"

    raise RuntimeError("No free drive letter found")


def create_subst_drive(path):
    drive = find_free_drive_letter()
    subprocess.run(["subst", drive, path], check=True)
    return drive


def remove_subst_drive(drive):
    subprocess.run(["subst", drive, "/d"], check=False)