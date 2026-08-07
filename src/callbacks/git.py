import subprocess


def save_git_hash(output_path):

    try:

        commit = subprocess.check_output(

            [
                "git",
                "rev-parse",
                "HEAD",
            ]

        ).decode().strip()

    except Exception:

        commit = "unknown"

    with open(
        output_path,
        "w",
    ) as f:

        f.write(commit)