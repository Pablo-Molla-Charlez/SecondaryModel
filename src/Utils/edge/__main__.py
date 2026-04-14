"""CLI entrypoint — ``python -m Utils.edge ...``.

Delegates to :func:`Utils.edge._impl.main`, preserving the argument surface
of the pre-refactor ``Utils/edge.py`` CLI.
"""
from Utils.edge.edge import main


if __name__ == "__main__":
    main()
