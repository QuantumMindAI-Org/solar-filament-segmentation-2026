#!/usr/bin/env python3
"""Run a notebook's code cells, in order, in one plain Python namespace (no Jupyter needed).

This stands in for "Restart and run all" where Jupyter is not installed. It reads the notebook JSON,
takes the code cells in order, joins each cell's source, and executes it with exec() in one shared
dictionary, so a name defined in an early cell is visible in a later one, as in a kernel.
It refuses what it cannot emulate: IPython magics ("%..." and "!..." lines). The last-expression echo
of a notebook cell is not reproduced; print() output is. The working folder is switched to the
notebook's folder for the run (the notebook assumes it is in the repository root) and restored after.

  python -B run_notebook_cells.py NOTEBOOK [--env NAME=VALUE ...]
Exit 0 when every code cell ran; 1 on the first failing cell (its number and the error are printed).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True


def load_notebook(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (isinstance(notebook, dict) and notebook.get("nbformat") == 4 and isinstance(notebook.get("cells"), list)):
        raise ValueError("NOT_NBFORMAT4")
    return notebook


def cell_source(cell):
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def code_cells(notebook):
    """[(cell index, source)] of the code cells, in notebook order."""
    return [(index, cell_source(cell)) for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"]


def check_no_magics(cells):
    for index, source in cells:
        for line in source.splitlines():
            if line.lstrip().startswith(("%", "!")):
                raise ValueError("MAGIC_IN_CELL_%d: %s" % (index, line.strip()))


def run_cells(notebook, folder, env=None, namespace=None):
    """Execute every code cell; returns the namespace. A failing cell raises RuntimeError naming it."""
    cells = code_cells(notebook)
    check_no_magics(cells)
    namespace = {"__name__": "__main__"} if namespace is None else namespace
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in (env or {})}
    saved_path = list(sys.path)
    try:
        os.chdir(folder)
        os.environ.update(env or {})
        for index, source in cells:
            try:
                exec(compile(source, "<cell %d>" % index, "exec"), namespace)
            except Exception as error:
                raise RuntimeError("CELL_%d_FAILED: %s: %s" % (index, type(error).__name__, error)) from error
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return namespace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE, repeatable")
    args = parser.parse_args(argv)
    env = dict(item.split("=", 1) for item in args.env)
    path = Path(args.notebook).absolute()
    try:
        run_cells(load_notebook(path), path.parent, env)
    except (ValueError, RuntimeError) as error:
        print("FAILED " + str(error), file=sys.stderr)
        if error.__cause__ is not None:
            traceback.print_exception(error.__cause__, file=sys.stderr)
        return 1
    print("ALL CELLS RAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
