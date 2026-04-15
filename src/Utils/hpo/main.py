"""CLI entrypoint — ``python -m Utils.hpo.main ...``.

Delegates to :func:`Utils.hpo._impl.main` which preserves the exact argument
surface of the pre-refactor ``Utils/HPO.py`` CLI.
"""
from Utils.hpo.runner import main


if __name__ == "__main__":
    main()
