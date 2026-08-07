import shutil


def backup_config(
    source,
    destination,
):

    shutil.copy2(
        source,
        destination,
    )