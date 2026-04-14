"""CLI entrypoint — ``python -m Utils.ocp.analysis ...``.

Delegates to :func:`Utils.ocp._analysis_impl.main`.
"""
from Utils.ocp._analysis_impl import main


if __name__ == "__main__":
    main()
