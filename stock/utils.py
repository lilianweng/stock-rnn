import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_path(rel_path):
    return os.path.join(REPO_ROOT, rel_path)
