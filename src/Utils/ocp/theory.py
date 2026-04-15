"""CLI entrypoint — ``python -m Utils.ocp.theory ...``.

Delegates to :func:`Utils.ocp._theory_impl.main`.
"""
from Utils.ocp._theory_impl import main

if __name__ == "__main__":
    main()
