#!/usr/bin/env python3
"""
Sprout - AI training snapshot manager (heads + borg)

Key concepts implemented:
- Active states are "heads". Heads point to run IDs and are represented by:
    WORK/.heads/<run_id>                   # active run folder (working copy)
    WORK/active/<head> -> .heads/<run_id>  # symlink for convenience
- Borg archives are named by run_id only: borg create <repo>::<run_id> <path>
- Metadata: WORK/.metadata.json with keys:
    {
      "runs": { "<run_id>": { group, parent, params, alias, description, created_at }, ... },
      "heads": { "<headname>": "<run_id>", ... }
    }

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Tuple, Optional
import argparse
import ast
import fcntl
import json
import os
import random
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import yaml

DEBUG = False

## Once Sprout is used for the first time, Borg complains with the
## following: "Warning: Attempting to access a previously unknown
## unencrypted repository!". Suppress that.
BORG_ENV = {'BORG_UNKNOWN_UNENCRYPTED_REPO_ACCESS_IS_OK': 'yes'}

# -------------------------
# Exceptions
# -------------------------

class SproutError(Exception):
    """Generic Sprout error for API callers."""
    pass


# -------------------------
# Helpers
# -------------------------

def now_iso() -> str:
    dt = datetime.now().astimezone()
    return dt.replace(microsecond=0).isoformat()

def to_iso(ts: str) -> str:
    dt = datetime.fromisoformat(ts)
    return dt.replace(microsecond=0).isoformat()

def sh_stream(cmd: str, env: Optional[dict] = {}) -> Optional[Iterator[str]]:
    """
    Run a shell command and stream output as it arrives.
    """

    if DEBUG:
        print(f"- {cmd}")

    env = os.environ | env

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,  # unbuffered
        text=True,
        env=env,
    )
    def _gen():
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            yield line.strip()

    return _gen

def sh(cmd: str,
       env: Optional[dict] = {},
       no_output: Optional[bool] = False,
       stream: Optional[bool] = False,) -> str:
    """Run a shell command. Exceptions are handled by a caller."""
    assert not (no_output and stream)
    if DEBUG:
        print(f"- {cmd}")
    env = os.environ | env
    if no_output:
        subprocess.check_call(cmd, shell=True, text=True, env=env)
        return None
    if stream:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # unbuffered
            text=True,
            env=env,
        )
        def _gen():
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                yield line.strip()
        return _gen()
    return subprocess.check_output(cmd, shell=True, text=True, env=env)

def rmtree(path: str) -> None:
    if DEBUG:
        print(f"- rm -rf {path}")
    shutil.rmtree(path)

def unlink(path: str) -> None:
    if DEBUG:
        print(f"- unlink {path}")
    os.unlink(path)

def mkdir_p(path: str) -> None:
    if DEBUG:
        print(f"- mkdir -p {path}")
    os.makedirs(path, exist_ok=True)

def cp_r(source: str, dist: str) -> None:
    if DEBUG:
        print(f"- cp -r {source} {dist}")
    shutil.copytree(source, dist, dirs_exist_ok=True)

def is_dir(path: str) -> bool:
    return os.path.isdir(path)

def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def random_sha8(rng: random.Random) -> str:
    return "".join(rng.choices(string.hexdigits.lower(), k=8))

def parse_params_string(params_str: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse params string using shlex. Return:
      - None if params_str is None (don't touch)
      - {} if params_str == "" (clear params)
      - dict otherwise
    """
    if params_str is None:
        return None
    s = params_str.strip()
    if s == "":
        return {}
    parts = shlex.split(s)
    # Split on possible `=`
    parts = [e for p in parts for e in p.split('=')]
    if len(parts) % 2 != 0:
        raise SproutError("params string must be key value pairs")
    out = {}
    for k, v in zip(parts[::2], parts[1::2]):
        try:
            v = ast.literal_eval(v)
        except:
            pass
        out[k.lstrip("-")] = v
    return out

def decode_escapes(s: Optional[str]) -> Optional[str]:
    """Convert escape sequences like '\\n' into real newlines."""
    if s is None:
        return None
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except Exception:
        return s

def extract_arg_defs(parser: argparse.ArgumentParser):
    """Extract definitions from argparse (not parsed values)"""
    arg_defs = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        arg_defs[action.dest] = {
            "default": action.default,
            "required": action.required,
            "option_strings": action.option_strings,
            "action": action,
        }
    return arg_defs

def make_cli_opts(arg_defs, new_params):
    """Compare new parameters with argparse defaults and required
    values, and return a list of CLI options"""
    opts = []
    missing = []

    for name, meta in arg_defs.items():
        default_val = meta["default"]
        required = meta["required"]
        new_val = new_params.get(name, None)

        # Handle required params: add with special marker if missing
        if required and new_val is None:
            opt_string = meta["option_strings"][0]
            if isinstance(default_val, bool):
                missing.append(f"{opt_string}-REQUIRED")
            else:
                missing.append(f"{opt_string} ${name.upper()}")

        # Include if new param overrides default
        elif new_val is not None and new_val != default_val:
            opt_string = meta["option_strings"][0]
            if isinstance(default_val, bool):
                if new_val:
                    opts.append(opt_string)
            else:
                opts.append(f"{opt_string} {new_val}")

        # Still output if params is required
        elif required:
            opt_string = meta["option_strings"][0]
            if isinstance(default_val, bool):
                if default_val:
                    opts.append(opt_string)
            else:
                opts.append(f"{opt_string} {default_val}")

    return opts + missing

def format_cli_opts(cli_opts, prefix=""):
    if not sys.stdout.isatty():
        width = 80
    else:
        width = shutil.get_terminal_size((80, 20)).columns

    lines, current = [], prefix
    for opt in cli_opts:
        piece = ("" if current == prefix else " ") + opt
        if len(current) + len(piece) > width:
            # close current line with backslash
            lines.append(current + " \\")
            current = prefix + opt
        else:
            current += piece
    if current:
        lines.append(current)

    return "\n".join(lines)

def color(text, rgb=(255, 255, 255), bold=False):
    """Colorize text with RGB truecolor (default: white)."""
    if not sys.stdout.isatty():
        return text
    codes = []
    if bold:
        codes.append("1")
    if rgb:
        r, g, b = rgb
        codes.append(f"38;2;{r};{g};{b}")
    prefix = f"\033[{';'.join(codes)}m" if codes else ""
    return f"{prefix}{text}\033[0m"

def flatten(d, prefix=""):
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, path)
        else:
            yield f"{path}: {v}"

# -------------------------
# File lock uses flock()
# -------------------------

class FileLock:
    def __init__(self, path):
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        self._refs = 0
        assert self._fd >= 0

    def __del__(self):
        os.close(self._fd)

    def acquire(self):
        if self._refs == 0:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._refs += 1

    def release(self):
        assert self._refs > 0
        self._refs -= 1
        if self._refs == 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

@contextmanager
def scoped_lock(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

def locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.lock.acquire()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.lock.release()
    return wrapper

# -------------------------
# Sprout API
# -------------------------

class Sprout:
    """
    API class for Sprout. Methods do not print.
    CLI wrapper should catch exceptions and print user-facing messages.
    """

    def __init__(self, working_path: str, seed: Optional[int] = None):
        # RNG for run id generation
        self.rng = random.Random(seed)
        # absolute working path (used for borg commands cd)
        self.working_path = os.path.abspath(working_path)

        # borg repo path (inside working folder)
        self.repo_path = os.path.join(self.working_path, ".borg")

        # heads path: user requested this be relative to the provided working path
        # (we keep it as a path joined with the original working argument, not absolute)
        self.heads_path = os.path.join(working_path, ".heads")

        # active path, where links to active heads are kept
        self.active_path = os.path.join(self.working_path, "active")

        # create directories
        mkdir_p(self.working_path)
        mkdir_p(self.heads_path)
        mkdir_p(self.active_path)

        # create a file lock
        self.lock = FileLock(os.path.join(working_path, ".lock"))

        with scoped_lock(self.lock):
            # initialize borg if needed
            if not is_dir(self.repo_path):
                self._borg_init()

            # metadata file inside repo
            self.meta_file = os.path.join(self.working_path, ".metadata.json")
            if not os.path.exists(self.meta_file):
                write_json(self.meta_file, {"runs": {}, "heads": {}})

    # -------------------------
    # Metadata helpers
    # -------------------------

    def _load_meta(self) -> dict:
        return read_json(self.meta_file)

    def _save_meta(self, meta: dict) -> None:
        shutil.copy(self.meta_file, f"{self.meta_file}.bkp")
        write_json(self.meta_file, meta)

    def _head_dir(self, run_id: str) -> str:
        """Path to the folder with active run contents (inside .heads)."""
        return os.path.join(self.heads_path, run_id)

    def _symlink_target_rel(self, run_id: str) -> str:
        """Relative symlink target from WORK/active/<head> to .heads/<run_id>."""
        # produce relative path from working_path
        rel = os.path.relpath(self._head_dir(run_id), start=self.active_path)
        return rel

    # -------------------------
    # Borg helpers
    # -------------------------

    def _borg_init(self) -> None:
        """Init a borg archive with --encryption none"""
        try:
            sh(f"borg init {shlex.quote(self.repo_path)} -e none", env=BORG_ENV)
        except subprocess.CalledProcessError:
            pass

    def _borg_repo_id(self) -> str:
        """Returns current repo ID"""
        try:
            out = sh(f"borg config {shlex.quote(self.repo_path)} id", env=BORG_ENV)
            return out.strip()
        except subprocess.CalledProcessError:
            return None

    def _borg_list(self) -> str:
        """List archives in borg archive, suppress non-error output"""
        try:
            return sh(f"borg list {shlex.quote(self.repo_path)} --error 2>/dev/null", env=BORG_ENV)
        except subprocess.CalledProcessError:
            return ""

    def _borg_delete(self, run_id: str) -> None:
        """Delete a borg archive if present, suppress non-error output"""
        try:
            sh(f"borg delete {shlex.quote(self.repo_path)}::{shlex.quote(run_id)} --force --error", env=BORG_ENV)
        except subprocess.CalledProcessError:
            # ignore if archive not present
            pass

    def _borg_delete_repo_cache(self) -> None:
        """Delete cache-only from borg's repo"""
        try:
            sh(f"borg delete --cache-only {shlex.quote(self.repo_path)}", env=BORG_ENV)
        except subprocess.CalledProcessError:
            # ignore if archive not present
            pass

    def _borg_create(self, run_id: str) -> None:
        """
        Create/overwrite borg archive <run_id> from .heads/<run_id> path.
        We cd into working_path and use a path relative to working_path to
        keep archive paths predictable.
        """
        # remove existing archive if any (suppress error)
        try:
            sh(f"borg delete {shlex.quote(self.repo_path)}::{shlex.quote(run_id)} --force --error", env=BORG_ENV)
        except subprocess.CalledProcessError:
            pass

        rel_archive_dir = os.path.join(".heads", run_id)

        # cd into working_path so extraction returns paths under working_path
        sh(f"cd {shlex.quote(self.working_path)} && borg create {shlex.quote(self.repo_path)}::{shlex.quote(run_id)} {shlex.quote(rel_archive_dir)}", env=BORG_ENV)

    def _borg_extract(self, run_id: str) -> None:
        """
        Extract a borg archive into working_path.
        We rely on borg to restore relative paths correctly.
        """
        # run in working_path so extraction restores files into the correct tree
        sh(f"cd {shlex.quote(self.working_path)} && borg extract {shlex.quote(self.repo_path)}::{shlex.quote(run_id)}", env=BORG_ENV)

    # -------------------------
    # Main methods
    # -------------------------

    def random_run_id(self) -> str:
        meta = self._load_meta()
        existing = set(meta.get("runs", {}).keys())
        while True:
            rid = random_sha8(self.rng)
            if rid not in existing:
                return rid

    def _snapshot_head(self, head: str, meta: dict,
                       params: Optional[Dict[str, str]] = None,
                       description_str: Optional[str] = None,
                       alias_str: Optional[str] = None) -> str:
        """
        Snapshot the current active head by creating a new immutable run (snap_id)
        which becomes the parent of the previous active run.

        Returns snapshot run id (snap_id).

        The operation:
          - read run_id for head from meta["heads"]
          - ensure .heads/<run_id> exists
          - create a new snap_id
          - create borg archive for snap_id from .heads/<run_id>
          - insert metadata: snap.parent <- old_parent, snap.params/alias/desc copied
          - update old_run.parent = snap_id
          - save metadata
        """
        heads = meta.get("heads", {})

        if head not in heads:
            raise SproutError(f"head '{head}' not found")

        run_id = heads[head]
        head_dir = self._head_dir(run_id)
        if not is_dir(head_dir):
            raise SproutError(f"active folder for run '{run_id}' not found at {head_dir}")

        # prepare snapshot id
        snap_id = self.random_run_id()
        snap_dir = self._head_dir(snap_id)
        mkdir_p(snap_dir)

        # copytree into empty newly created snapshot
        cp_r(head_dir, snap_dir)

        # create borg archive for snapshot from active folder
        self._borg_create(snap_id)

        # Delete dir of newly created snapshot once archived
        rmtree(snap_dir)

        # insert metadata snapshot (copy fields)
        runs = meta.setdefault("runs", {})
        old_run = runs.get(run_id)
        if old_run is None:
            raise SproutError(f"run '{run_id}' not present in metadata")

        snap_entry = {
            "group": old_run.get("group"),
            "parent": old_run.get("parent"),
            "params": old_run.get("params", {}),
            "alias": old_run.get("alias", ""),
            "description": old_run.get("description", ""),
            "custom": old_run.get("custom", {}),
            "created_at": now_iso()
        }
        runs[snap_id] = snap_entry

        # update old run parent -> snap_id (insert snapshot above the active run)
        old_run["parent"] = snap_id
        if params is not None:
            old_run["params"] = params
        if description_str is not None:
            old_run["description"] = description_str
        if alias_str is not None:
            old_run["alias"] = alias_str

        # save metadata
        self._save_meta(meta)

        return snap_id

    @locked
    def create(self,
               group: Optional[str] = None,
               head: Optional[str] = None,
               from_run: Optional[str] = None,
               from_head: Optional[str] = None,
               params_str: Optional[str] = None,
               description_str: Optional[str] = None,
               alias_str: Optional[str] = None) -> str:
        """
        Create a new run.

        Behavior:
          - If from_head is provided and head is NOT provided, create acts as a
            snapshot operation only: it snapshots the active head (inserts a node)
            and returns the snapshot id. (This matches your tests.)
          - Otherwise, it creates a new run whose parent is:
              * if from_run provided: from_run (if that run is active it will be snapshotted first,
                and parent becomes the snapshot id),
              * if from_head provided: snapshot of that head (head is snapshotted first),
              * if neither: None (new root run).
            After creating the new run, if head is provided we create an active folder
            and symlink WORK/active/<head> -> .heads/<run_id>. If head is omitted we persist-only
            (archive and remove local folder).
        Returns newly created run id (or snapshot id if snapshot-only).
        """
        if from_run and from_head:
            raise SproutError("either run or head must be provided, but not both")
        if group and (from_run or from_head):
            raise SproutError("either group or run/head must be provided, but not both")
        if not (group or from_run or from_head):
            raise SproutError("group or run or head must be provided")

        meta = self._load_meta()
        runs = meta.setdefault("runs", {})
        heads = meta.setdefault("heads", {})

        if head in heads:
            raise SproutError(f"head {head} already exists")

        # determine source run (if any)
        src_run = None
        if from_head:
            if from_head not in heads:
                raise SproutError(f"head '{from_head}' not found")
            src_run = heads[from_head]
        elif from_run:
            src_run = from_run

        if src_run and src_run not in runs:
            raise SproutError(f"run '{src_run}' not found")

        # parse params/description/alias
        # params parsing returns None if not provided
        params = parse_params_string(params_str)
        description = decode_escapes(description_str) if description_str is not None else None
        custom_dict = {}

        # if the source run exists and is active (any head points to it), snapshot it first
        parent_for_new = None
        if src_run:
            # is src_run active? check if there is a head mapping to it
            active_heads = [h for h, rid in heads.items() if rid == src_run]
            if active_heads:
                # snapshot the active run (insert snapshot above it)
                if not head:
                    # If src run is provided AND no target head
                    # specified treat as snapshot-only and provide new
                    # params
                    snap_id = self._snapshot_head(active_heads[0], meta,
                                                  params=params,
                                                  description_str=description,
                                                  alias_str=alias_str)
                    return snap_id

                snap_id = self._snapshot_head(active_heads[0], meta)
                parent_for_new = snap_id
            else:
                parent_for_new = src_run

            group = runs[src_run]['group']

        # allocate new run id
        new_run_id = self.random_run_id()

        # create target head folder (under .heads)
        target_dir = self._head_dir(new_run_id)
        mkdir_p(target_dir)

        # if parent exists, we need to copy its folder content into target_dir
        if parent_for_new:
            # if parent has an active folder, copy from it
            parent_active_dir = self._head_dir(parent_for_new)
            if is_dir(parent_active_dir):
                # copytree into empty target_dir
                cp_r(parent_active_dir, target_dir)
            else:
                # parent not active -> extract from borg then copy
                # extract into working tree (borg extract will create a folder under working_path)
                self._borg_extract(parent_for_new)
                try:
                    cp_r(parent_active_dir, target_dir)
                finally:
                    # remove extracted temporary folder if it exists in .heads
                    rmtree(parent_active_dir)

            # Inherit a few fields from parent
            parent_run = runs[parent_for_new]
            if params is None:
                params = parent_run["params"]
            if alias_str is None:
                alias_str = parent_run["alias"]
            if description is None:
                description = parent_run["description"]
            custom_dict = parent_run.get("custom", {})

        # create borg archive for the new run (archive name is run id)
        self._borg_create(new_run_id)

        # if no head requested, remove the local folder (persist-only)
        if not head:
            if is_dir(target_dir):
                rmtree(target_dir)
        else:
            # create/update symlink WORK/active/<head> -> .heads/<new_run_id>
            symlink_path = os.path.join(self.active_path, head)
            rel_target = self._symlink_target_rel(new_run_id)
            # remove existing symlink
            if os.path.islink(symlink_path):
                try:
                    unlink(symlink_path)
                except OSError:
                    pass
            os.symlink(rel_target, symlink_path)
            # update heads mapping
            heads[head] = new_run_id

        # persist metadata for new run (store even if persisted-only)
        runs[new_run_id] = {
            "group": group,
            "parent": parent_for_new,
            "params": params or {},
            "alias": alias_str or "",
            "description": description or "",
            "custom": custom_dict,
            "created_at": now_iso()
        }
        self._save_meta(meta)

        return new_run_id

    @locked
    def persist(self, head: Optional[str] = None) -> List[str]:
        """
        Persist a single head or all heads.
        Returns list of persisted run ids.

        Persisting recreates borg archive for the active folder and deactivates the head
        (removes symlink and .heads/<run_id> folder) unless other heads point to same run.
        """
        meta = self._load_meta()
        heads = meta.get("heads", {})

        to_process = []
        if head:
            if head not in heads:
                raise SproutError(f"head '{head}' not found")
            to_process = [head]
        else:
            to_process = sorted(heads.keys())

        persisted = []

        for h in to_process:
            run_id = heads[h]
            head_dir = self._head_dir(run_id)
            if not is_dir(head_dir):
                raise SproutError(f"head '{h}' has no active folder at {head_dir}")

            # recreate borg archive from head_dir
            self._borg_create(run_id)

            # remove symlink WORK/active/<head>
            symlink_path = os.path.join(self.active_path, h)
            if os.path.islink(symlink_path):
                try:
                    unlink(symlink_path)
                except OSError:
                    pass

            # remove the head folder if no other heads point to the same run
            meta = self._load_meta()  # reload to be safe
            other_heads = [k for k, v in meta.get("heads", {}).items() if v == run_id and k != h]
            if not other_heads:
                if is_dir(head_dir):
                    rmtree(head_dir)

            # remove the head mapping
            meta.get("heads", {}).pop(h, None)
            self._save_meta(meta)

            persisted.append(run_id)

        return persisted

    def remove_run_recursive(self, run_id: str, run: dict, meta: dict, whole_branch: bool = False) -> None:
        """
        Remove run and optionally its children. Mutates meta but does not save it.
        """
        runs = meta.get("runs", {})
        heads = meta.get("heads", {})
        parent = run.get("parent")

        # find children
        children = [(rid, r) for rid, r in runs.items() if r.get("parent") == run_id]
        for ch_rid, ch in children:
            if whole_branch:
                # recursively remove children
                self.remove_run_recursive(ch_rid, ch, meta, whole_branch=True)
            else:
                # reparent children
                ch["parent"] = parent

        # remove heads mapping and folders pointing to this run
        heads_pointing = [h for h, rid in heads.items() if rid == run_id]
        for h in heads_pointing:
            symlink_path = os.path.join(self.active_path, h)
            if os.path.islink(symlink_path):
                try:
                    unlink(symlink_path)
                except OSError:
                    pass
            heads.pop(h, None)

        # delete borg archive (ignore errors)
        try:
            self._borg_delete(run_id)
        except Exception:
            pass

        # remove head dir if exists
        head_dir = self._head_dir(run_id)
        if is_dir(head_dir):
            rmtree(head_dir)

        # remove metadata entry
        runs.pop(run_id, None)

    @locked
    def remove(self,
               group: Optional[str] = None,
               run: Optional[str] = None,
               head: Optional[str] = None,
               whole_branch: bool = False) -> dict:
        """
        Remove:
          - group: all runs for the group,
          - run: a specific run id,
          - head: a head name (deactivate and remove active folder).
        Returns a dict describing removed items.
        """
        if run and head:
            raise SproutError("either run or head must be provided, but not both")
        if group and (run or head):
            raise SproutError("either group or run/head must be provided, but not both")
        if not (group or run or head):
            raise SproutError("group or run or head must be provided")

        meta = self._load_meta()
        runs = meta.get("runs", {})
        heads = meta.get("heads", {})

        if group:
            to_delete = [(rid, r) for rid, r in runs.items() if r.get("group") == group]
            if not to_delete:
                raise SproutError(f"group '{group}' not found")
            for rid, r in to_delete:
                self.remove_run_recursive(rid, r, meta, whole_branch=True)
            self._save_meta(meta)
            return to_delete

        if head:
            if head not in heads:
                raise SproutError(f"head '{head}' not found")
            run = heads[head]

        r = runs.get(run, {})
        if not r:
            raise SproutError(f"run '{run}' not found")

        self.remove_run_recursive(run, r, meta, whole_branch=whole_branch)
        self._save_meta(meta)
        return [run]

    @locked
    def edit(self,
             run: Optional[str] = None,
             head: Optional[str] = None,
             params_str: Optional[str] = None,
             description_str: Optional[str] = None,
             alias_str: Optional[str] = None,
             created_str: Optional[str] = None,
             custom_dict: Optional[dict] = None) -> str:
        """
        Edit metadata for a run. None means don't touch. Empty string clears.
        """
        meta = self._load_meta()
        run, run_id = self.get_run(run=run, head=head, meta=meta)

        parsed = parse_params_string(params_str)
        if parsed is not None:
            run["params"] = parsed

        if description_str is not None:
            run["description"] = decode_escapes(description_str)

        if alias_str is not None:
            run["alias"] = alias_str

        if created_str is not None:
            run["created_at"] = created_str

        if custom_dict is not None:
            run["custom"] = custom_dict

        self._save_meta(meta)
        return run_id

    @locked
    def get_run(self,
                run: Optional[str] = None,
                head: Optional[str] = None,
                meta: Optional[dict] = None) -> Tuple[Dict[str, dict], str]:
        """
        Return (run, run_id) by run ID or head name
        """
        if meta is None:
            meta = self._load_meta()
        runs = meta.get("runs", {})
        heads = meta.get("heads", {})

        if run and head:
            raise SproutError("either run or head must be provided, but not both")
        if not (run or head):
            raise SproutError("run or head must be provided")

        run_id = None
        if run:
            run_id = run
        else:
            if head not in heads:
                raise SproutError(f"head '{head}' not found")
            run_id = heads[head]

        if run_id not in runs:
            raise SproutError(f"run '{run_id}' not found")

        return runs[run_id], run_id

    @locked
    def get_tree(self, group: Optional[str] = None) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, List[str]]]:
        """
        Return (runs, heads, tree) where:
          - runs is dict run_id -> run_metadata
          - heads is dict head -> run_id
          - tree is adjacency dict run_id -> [child_ids...]
        If group provided, runs filtered to that group.
        """
        meta = self._load_meta()
        all_runs = meta.get("runs", {})
        heads = meta.get("heads", {})
        if group:
            runs = {rid: r for rid, r in all_runs.items() if r.get("group") == group}
        else:
            runs = all_runs

        tree: Dict[str, List[str]] = {rid: [] for rid in runs.keys()}
        for rid, r in runs.items():
            parent = r.get("parent")
            if parent and parent in runs:
                tree[parent].append(rid)
        return runs, heads, tree

    @locked
    def history_chain(self, run_id: str) -> List[Tuple[str, dict]]:
        meta = self._load_meta()
        runs = meta.get("runs", {})
        if run_id not in runs:
            raise SproutError(f"run '{run_id}' not found")
        chain: List[Tuple[str, dict]] = []
        cur = run_id
        while cur:
            r = runs.get(cur)
            if not r:
                break
            chain.append((cur, r))
            cur = r.get("parent")
        chain.reverse()
        return chain

    @locked
    def verify(self) -> None:
        """
        Verify consistency between borg repo, metadata.json, and working directory.

        Checks:
          1. Borg archives match runs in metadata.
          2. .heads folders correspond to runs and are properly listed as heads.
          3. Symlinks in working dir point to .heads and match count.
        Raises RuntimeError on mismatch.
        """
        meta = self._load_meta()
        runs = meta.get("runs", {})
        heads = meta.get("heads", {})

        # 1. Check borg archives
        out = self._borg_list()
        borg_ids = {line.split()[0] for line in out.strip().splitlines()}
        meta_ids = set(runs.keys())
        if borg_ids != meta_ids:
            raise RuntimeError(
                f"[verify] mismatch between borg archives and metadata:\n"
                f"borg-only: {borg_ids - meta_ids}\n"
                f"meta-only: {meta_ids - borg_ids}"
            )

        # 2. Check .heads folders
        head_dirs = {d for d in os.listdir(self.heads_path) if is_dir(os.path.join(self.heads_path, d))}
        meta_heads = set(heads.values())
        if not head_dirs.issubset(meta_ids):
            raise RuntimeError(f"[verify] unknown run dirs in .heads: {head_dirs - meta_ids}")
        if head_dirs != meta_heads:
            raise RuntimeError(
                f"[verify] mismatch between .heads dirs and heads in metadata:\n"
                f".heads-only: {head_dirs - meta_heads}\n"
                f"meta-only: {meta_heads - head_dirs}"
            )

        # 3. Check symlinks in working_path
        symlinks = {
            name: os.readlink(os.path.join(self.active_path, name))
            for name in os.listdir(self.active_path)
            if os.path.islink(os.path.join(self.active_path, name))
        }

        for name, target in symlinks.items():
            if not target.startswith("../.heads/"):
                raise RuntimeError(f"[verify] symlink {name} points outside .heads: {target}")
            target_id = os.path.basename(target)
            if target_id not in head_dirs:
                raise RuntimeError(f"[verify] symlink {name} points to missing head dir {target_id}")

        if len(symlinks) != len(head_dirs):
            raise RuntimeError(
                f"[verify] number of symlinks {len(symlinks)} != number of .heads dirs {len(head_dirs)}"
            )

        if set(symlinks.keys()) != set(heads.keys()):
            raise RuntimeError(
                f"[verify] symlinks {set(symlinks.keys())} != heads {set(heads.keys())}"
            )


# -------------------------
# Small CLI wrapper
# -------------------------

def cli_create(args, sprout: Sprout) -> int:
    try:
        run_id = sprout.create(group=args.group,
                               head=args.head,
                               from_run=args.from_run,
                               from_head=args.from_head,
                               params_str=args.params,
                               description_str=args.description,
                               alias_str=args.alias)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if run_id:
        if args.head:
            print(f"Created run {run_id} and head '{args.head}' -> .heads/{run_id}")
        else:
            print(f"Created run {run_id}")
    return 0


def cli_persist(args, sprout: Sprout) -> int:
    try:
        persisted = sprout.persist(head=args.head)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if persisted:
        for rid in persisted:
            print(f"Persisted {rid}")
    else:
        print("Nothing to persist")
    return 0


def cli_remove(args, sprout: Sprout) -> int:
    try:
        if args.group:
            _ = sprout.remove(group=args.group, whole_branch=args.whole_branch)
            print(f"Removed group {args.group}")
        elif args.run:
            _ = sprout.remove(run=args.run, whole_branch=args.whole_branch)
            print(f"Removed run {args.run}")
        elif args.head:
            _ = sprout.remove(head=args.head)
            print(f"Removed head {args.head}")
        else:
            print("ERROR: remove requires --group, --run, or --head", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0

def cli_edit(args, sprout: Sprout,
             default_parser: Optional[argparse.ArgumentParser] = None) -> int:
    try:
        meta = sprout._load_meta()

        alias_str = args.alias
        params_str = args.params
        description_str = args.description
        created_str = None
        custom_dict = None

        if not (params_str or alias_str or description_str):
            description_template = ["# Enter description here", "# Keep indentation"]

            run, _ = sprout.get_run(run=args.run, head=args.head, meta=meta)

            # Prepare params
            params = run.get("params", {})
            max_len = 0
            for k, v in params.items():
                max_len = max(max_len, len(k))
            fmt = "    %s%s%s"
            params_list = []
            for k, v in params.items():
                params_list.append(fmt % (k, " " * (max_len - len(k)), " = " + repr(v)))
            params_str = "\n".join(params_list)

            # Prepare CLI opts
            opts_str = ""
            if default_parser:
                arg_defs = extract_arg_defs(default_parser)
                cli_opts = make_cli_opts(arg_defs, params)
                opts_str = format_cli_opts(cli_opts, prefix="# ")

            # Prepare description
            description = run.get("description", "").rstrip()
            if not description:
                description = "\n".join(description_template)
            description = textwrap.indent(description, " " * 4)

            # Prepare created
            created = run.get("created_at", "") or ""
            if created:
                created = to_iso(created)

            # Prepare alias
            alias = run.get("alias", "") or ""

            # Prepare custom
            custom_str = ""
            custom_dict = run.get("custom", {}) or {}
            if custom_dict:
                custom_yaml = yaml.dump(custom_dict, indent=4)
                custom_str = "\n\n# CUSTOM FIELDS\n\n" + custom_yaml

            # Content
            content = \
f"""{opts_str}

created: {created}
alias: {alias}
params: |
{params_str}

description: |
{description}{custom_str}
"""

            # Write content
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
                tf.write(content)
                tf.flush()
                tmp = os.path.abspath(tf.name)

            # Open editor
            try:
                editor = os.environ.get("EDITOR", "vim")
                sh(f"{editor} {shlex.quote(tmp)}", no_output=True)
            except subprocess.CalledProcessError as e:
                os.unlink(tmp)
                print(f"ERROR: editor failed: {e}", file=sys.stderr)
                return 2

            # Read back
            with open(tmp, "r") as f:
                edited = f.read()

            os.unlink(tmp)

            # Strip CLI comment lines
            cleaned = "\n".join(line for line in edited.splitlines() if not line.startswith("#"))

            # Parse YAML back
            try:
                data = yaml.safe_load(cleaned)
                if not data:
                    data = {}
            except yaml.YAMLError as e:
                print(f"ERROR: YAML parsing error: {e}", file=sys.stderr)
                return 2

            # Update variables
            alias_str = data.get("alias", "") or ""
            created_str = data.get("created", "") or ""
            if created_str:
                created_str = to_iso(str(created_str))

            # Handle params
            params_str = data.get("params", "") or ""
            # Split into lines, drop empty ones
            params_lines = [line.strip() for line in params_str.splitlines() if line.strip()]

            # Split each line on '=', strip both sides, reformat
            pairs = []
            for line in params_lines:
                if "=" in line:
                    k, v = line.split("=", 1)  # split only once
                    pairs.append(f"{k.strip()}={v.strip()}")
            # Join into a single string
            params_str = " ".join(pairs)

            # Handle description
            description_str = data.get("description", "") or ""
            description_str = description_str.rstrip()
            if description_str.strip() == "\n".join(description_template):
                # strip away if unchanged template
                description_str = ""

            # Handle custom
            known_keys = ['created', 'alias', 'params', 'description']
            custom_keys = data.keys() - set(known_keys)
            custom_dict = {k: data[k] for k in custom_keys}

        sprout.edit(run=args.run,
                    head=args.head,
                    params_str=params_str,
                    description_str=description_str,
                    alias_str=alias_str,
                    created_str=created_str,
                    custom_dict=custom_dict)

        if args.run:
            print(f"Edited run {args.run}")
        else:
            print(f"Edited head {args.head}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cli_tree(args, sprout: Sprout) -> int:
    try:
        runs, heads, tree = sprout.get_tree(group=args.group)

        # compute roots
        all_children = {c for children in tree.values() for c in children}
        roots = [rid for rid in runs.keys() if rid not in all_children]

        # reverse mapping run_id -> [heads...]
        run_to_heads = defaultdict(list)
        for h, rid in heads.items():
            run_to_heads[rid].append(h)

        def diff_params_and_custom_dict(r: dict, parent_r: dict, prefix: str,
                                        has_children: bool, has_siblings: bool):
            prefix = "\n" + prefix
            def tree_prefix(ch):
                if has_siblings and not has_children:
                    return f"│    {ch} "
                if has_siblings and has_children:
                    return f"│  │ {ch} "
                if not has_siblings and has_children:
                    return f"   │ {ch} "
                return f"     {ch} "

            diffs = []
            out_str = "[all params]"
            if parent_r:
                params = parent_r["params"]
                for k, v in r["params"].items():
                    if k not in params:
                        diffs.append(f"{k}:{v}")
                    else:
                        old = params[k]
                        if old != v:
                            diffs.append(f"{k}: {old} -> {v}")
                if not r["params"]:
                    out_str = "∅"
                elif not diffs:
                    out_str = "Δ ∅"
                else:
                    params_prefix = prefix + tree_prefix("⇾")
                    params_str = params_prefix + params_prefix.join(diffs)
                    out_str = params_str

            custom_dict = r.get("custom", {}) or {}
            if custom_dict:
                custom_list = list(flatten(custom_dict))
                custom_prefix = prefix + tree_prefix("≡")
                custom_str = custom_prefix + custom_prefix.join(custom_list)
                out_str += custom_str

            return out_str

        def print_node(rid: str, prefix: str, is_last: bool) -> None:
            r = runs[rid]
            children = tree.get(rid, [])
            parent_r = runs[r["parent"]] if r["parent"] else None
            headnames = run_to_heads.get(rid, [])
            alias = f" ({r.get('alias')})" if r.get("alias") else ""
            if args.verbose:
                params_str = json.dumps(r.get("params", {}))
                ts = " " + to_iso(r["created_at"])
            else:
                params_str = diff_params_and_custom_dict(r, parent_r, prefix,
                                                         bool(children), not is_last)
                ts = ""

            branch = "└─" if is_last else "├─"

            if headnames:
                # light green
                dot = color("●", rgb=(0, 255, 0), bold=True)
                heads_display = " ".join([color(h, bold=True) for h in headnames])
                line = f"{prefix}{branch} {dot} {heads_display}{alias} {params_str}{ts}"
            else:
                rid_display = color(rid)
                line = f"{prefix}{branch} {rid_display}{alias} {params_str}{ts}"

            print(line)

            new_prefix = prefix + ("   " if is_last else "│  ")
            for i, c in enumerate(children):
                is_last = i == (len(children) - 1)
                print_node(c, new_prefix, is_last)

        # group by group and print
        groups = sorted({r["group"] for r in runs.values()})
        for ig, g in enumerate(groups):
            if ig:
                print()
            group_display = color(f"▶ {g}", bold=True)
            print(group_display)
            roots_for_group = [rid for rid in roots if runs[rid]["group"] == g]
            for i, root in enumerate(roots_for_group):
                is_last = i == (len(roots_for_group) - 1)
                print_node(root, "", is_last)

        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cli_log(args,
            sprout: Sprout,
            default_parser: Optional[argparse.ArgumentParser] = None) -> int:

    arg_defs = None
    if default_parser:
        arg_defs = extract_arg_defs(default_parser)

    # renamed history: show ancestry diffs for run or head
    try:
        _, heads, _ = sprout.get_tree()
        run_id = None
        if args.run:
            run_id = args.run
        elif args.head:
            if args.head not in heads:
                print(f"ERROR: head '{args.head}' not found", file=sys.stderr)
                return 2
            run_id = heads[args.head]
        else:
            print("ERROR: log requires --run or --head", file=sys.stderr)
            return 2

        chain = sprout.history_chain(run_id)
        params: Dict[str, str] = {}
        for i, (rid, r) in enumerate(chain):
            run_params = r["params"]
            diffs = {}
            for k, v in run_params.items():
                if k not in params:
                    diffs[k] = repr(v)
                else:
                    old = params[k]
                    if old != v:
                        diffs[k] = f"{old} -> {v}"
            params.update(run_params)
            alias = f" ({r['alias']})" if r.get("alias") else ""
            ts = to_iso(r["created_at"])
            active_heads = [h for h, run in heads.items() if run == rid]
            if active_heads:
                rid_or_head = active_heads[0]
            else:
                rid_or_head = rid
            title = f"> {rid_or_head}{alias} at {ts}"
            if i > 0:
                print()
            print(color(title, bold=True))
            if arg_defs:
                cli_opts = make_cli_opts(arg_defs, run_params)
                opts = format_cli_opts(cli_opts, prefix="  ")
                print(f"{opts}")
                print()
            if i == 0:
                if diffs:
                    print("  Initial params:")
                else:
                    print("  No params")
            else:
                if diffs:
                    print("  Params changed:")
                else:
                    print("  No params changes")
            max_len = 0
            for k, v in diffs.items():
                max_len = max(max_len, len(k))
            fmt = "    %s%s%s"
            for k, v in diffs.items():
                print(fmt % (k, " " * (max_len - len(k)), " = " + v))

            description = r["description"]
            if description:
                description = textwrap.indent(description, " " * 4)
                print()
                print("  Description:")
                print(description)

            custom_dict = r.get("custom", {}) or {}
            if custom_dict:
                custom_yaml = yaml.dump(custom_dict, indent=4)
                custom_yaml = textwrap.indent(custom_yaml, " " * 4)
                print()
                print("  Custom fields:")
                print(custom_yaml)

        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cli_fetch(args, sprout: Sprout):

    src_host = args.src_host
    if not src_host.endswith("/"):
        src_host += "/"

    repo_id = sprout._borg_repo_id()
    if repo_id:
        # Prevent this fatal error: "Cache is newer than repository -
        # do you have multiple, independently updated repos with same
        # ID?" with the procedure taken from here:
        # https://borgbackup.readthedocs.io/en/stable/faq.html#this-is-either-an-attack-or-unsafe-warning
        sprout._borg_delete_repo_cache()
        sh(f"rm -f ~/.config/borg/security/{repo_id}/manifest-timestamp")

    cmd = f"rsync -avz --delete --ignore-errors --inplace --partial {src_host} {args.working} || true"
    for out in sh(cmd, stream=True):
        print(out)

def build_parser(prog, suppress_working_dir=False, add_help=True):
    parser = argparse.ArgumentParser(prog=prog, description="Sprout CLI", add_help=add_help)
    if suppress_working_dir:
        # Do not require --working and suppress help for it, expecting
        # it will be passed anyway in main()
        parser.add_argument("--working", help=argparse.SUPPRESS)
    else:
        parser.add_argument("--working", required=True, help="Sprout working directory")
    parser.add_argument("--debug", action="store_true", help="Enables debug output")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # create
    pc = sub.add_parser("create")
    pc.add_argument("group", nargs="?", help="Group name (required unless --from-run/--from-head used)")
    pc.add_argument("--head", help="Create an active head pointing to the new run")
    pc.add_argument("--from-run", dest="from_run", help="Copy from existing run id")
    pc.add_argument("--from-head", dest="from_head", help="Copy from an existing head name")
    pc.add_argument("--params", default=None, help="Set params: e.g. --params \"--lr 0.1 --batch 64\"")
    pc.add_argument("--description", default=None, help="Set description string (use bash $'line\\nline' for newlines)")
    pc.add_argument("--alias", default=None, help="Set alias")

    # persist
    pp = sub.add_parser("persist")
    pp.add_argument("head", nargs="?", help="Persist a head or all heads")

    # remove
    pr = sub.add_parser("remove")
    pr.add_argument("--group", help="Remove whole group and all runs")
    pr.add_argument("--run", dest="run", help="Remove run id")
    pr.add_argument("--head", help="Remove head and active state")
    pr.add_argument("--whole-branch", action="store_true", help="Removes the entire branch along with its children")

    # edit
    pe = sub.add_parser("edit")
    group = pe.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Run id to edit")
    group.add_argument("--head", help="Head name to edit")
    pe.add_argument("--params", default=None, help="Set params")
    pe.add_argument("--description", default=None, help="Set description")
    pe.add_argument("--alias", default=None, help="Set alias")

    # tree
    pt = sub.add_parser("tree", help="Show tree of runs (heads marked green)")
    pt.add_argument("--group", help="Filter by group")
    pt.add_argument("--verbose", action="store_true", help="Show full params and timestamps")

    # log (history)
    pl = sub.add_parser("log", help="Show ancestry/diffs (history) for run or head")
    pl.add_argument("--run", help="Run id")
    pl.add_argument("--head", help="Head name")

    # fetch
    prs = sub.add_parser("fetch", help="Fetches run states and heads from the remote repo")
    prs.add_argument("src_host", help="The remote source host for copying, can be specified in the user@host:/path format")

    return parser

# -------------------------
# CLI entrypoint
# -------------------------

def main(argv=None, default_parser: Optional[argparse.ArgumentParser] = None) -> int:
    parser = build_parser("sprout")
    args = parser.parse_args(argv)

    global DEBUG
    DEBUG = args.debug

    seed = os.environ.get("SEED")
    if seed:
        seed = int(seed)

    try:
        sprout = Sprout(args.working, seed=seed)
    except Exception as e:
        print(f"ERROR initializing Sprout: {e}", file=sys.stderr)
        return 2

    ret = None

    with scoped_lock(sprout.lock):
        if args.cmd == "create":
            ret = cli_create(args, sprout)
        elif args.cmd == "persist":
            ret = cli_persist(args, sprout)
        elif args.cmd == "remove":
            ret = cli_remove(args, sprout)
        elif args.cmd == "edit":
            ret = cli_edit(args, sprout, default_parser=default_parser)
        elif args.cmd == "tree":
            ret = cli_tree(args, sprout)
        elif args.cmd == "log":
            ret = cli_log(args, sprout, default_parser=default_parser)
        elif args.cmd == "fetch":
            ret = cli_fetch(args, sprout)
        else:
            parser.print_help()
            return 1
        sprout.verify()

    return ret

class SproutCLITests(unittest.TestCase):
    def setUp(self):
        # create a fresh temporary working directory
        self.tmpdir = tempfile.mkdtemp(prefix="sprout-test-")
        self.sprout = Sprout(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def run_cmd(self, args):
        """Run a shell command and return (returncode, stdout, stderr)."""
        result = subprocess.run(
            args,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def run_sprout(self, args):
        """Run sprout CLI command and return (returncode, stdout, stderr)."""
        return self.run_cmd(f"SEED=1 python3 {__file__} --working {self.tmpdir} " + args)

    def test_create_remove(self):
        # create group
        rc, out, err = self.run_sprout("create GroupA")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': []})
        self.assertEqual(heads, {})

        # create another run with a head
        rc, out, err = self.run_sprout("create GroupA --head A")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': []})
        self.assertEqual(heads, {'A': '20c9a09f'})

        # create head B from last run
        rc, out, err = self.run_sprout("create --head B --from-run 2ca5a9eb")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8'], '20c9a09f': [], '5ed00be8': []})
        self.assertEqual(heads, {'A': '20c9a09f', 'B': '5ed00be8'})

        # persist all
        rc, out, err = self.run_sprout("persist")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8'], '20c9a09f': [], '5ed00be8': []})
        self.assertEqual(heads, {})

        # create head A from last run
        rc, out, err = self.run_sprout("create --head A --from-run 2ca5a9eb")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': []})
        self.assertEqual(heads, {'A': '49049a55'})

        # create random file in A and checksum it
        rc, out, err = self.run_cmd(f"dd if=/dev/random of={self.tmpdir}/active/A/FILE1 bs=1M count=128")
        self.assertEqual(rc, 0, msg=err)
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum FILE1 >> md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "md5sum.txt"])

        # persist A
        rc, out, err = self.run_sprout("persist A")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': []})
        self.assertEqual(heads, {})

        # create head A from last run
        rc, out, err = self.run_sprout("create --head A --from-run 49049a55")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['4a60cce4'], '4a60cce4': []})
        self.assertEqual(heads, {'A': '4a60cce4'})

        # verify checksums
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1\n", msg=err)

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ ● A params=∅
        └─ 20c9a09f params=∅
        """

        # create head B from head A
        rc, out, err = self.run_sprout("create --head B --from-head A")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': [], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': []})
        self.assertEqual(heads, {'A': '4a60cce4', 'B': 'ce6cdcbc'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ ● A params=∅
        │        └─ ● B params=∅
        └─ 20c9a09f params=∅
        """

        # create random file in B and checksum it
        rc, out, err = self.run_cmd(f"dd if=/dev/random of={self.tmpdir}/active/B/FILE2 bs=1M count=128")
        self.assertEqual(rc, 0, msg=err)
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/B && md5sum FILE2 >> md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/B")), ["FILE1", "FILE2", "md5sum.txt"])

        # persist all
        rc, out, err = self.run_sprout("persist")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': [], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': []})
        self.assertEqual(heads, {})

        # create head A from run
        rc, out, err = self.run_sprout("create --head A --from-run 4a60cce4")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': ['05b93cfe'], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': [], '05b93cfe': []})
        self.assertEqual(heads, {'A': '05b93cfe'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ 4a60cce4 params=∅
        │        │  └─ ● A params=∅
        │        └─ ce6cdcbc params=∅
        └─ 20c9a09f params=∅
        """

        # create head B from run
        rc, out, err = self.run_sprout("create --head B --from-run ce6cdcbc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': ['05b93cfe'], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': ['89bbb8a0'], '05b93cfe': [], '89bbb8a0': []})
        self.assertEqual(heads, {'A': '05b93cfe', 'B': '89bbb8a0'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ 4a60cce4 params=∅
        │        │  └─ ● A params=∅
        │        └─ ce6cdcbc params=∅
        │           └─ ● B params=∅
        └─ 20c9a09f params=∅
        """

        # verify checksums for A
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1\n", msg=err)

        # verify checksums for B
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/B && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/B")), ["FILE1", "FILE2", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/B && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1 FILE2\n", msg=err)

        # create head B which already exists
        rc, out, err = self.run_sprout("create --head B --from-run 2ca5a9eb")
        self.assertEqual(rc, 2, msg=err)

        # create head C from a non-existent run
        rc, out, err = self.run_sprout("create --head C --from-run fefefefe")
        self.assertEqual(rc, 2, msg=err)

        # create from a non-existent head
        rc, out, err = self.run_sprout("create --from-head C")
        self.assertEqual(rc, 2, msg=err)

        # create random file in A and checksum it
        rc, out, err = self.run_cmd(f"dd if=/dev/random of={self.tmpdir}/active/A/FILE3 bs=1M count=128")
        self.assertEqual(rc, 0, msg=err)
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum FILE3 >> md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "FILE3", "md5sum.txt"])

        # get checksums from A
        rc, md5sum_A1, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt")
        self.assertEqual(rc, 0, msg=err)

        # move head A forward
        rc, out, err = self.run_sprout("create --from-head A")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': ['0ffd83bf'], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': ['89bbb8a0'], '05b93cfe': [], '89bbb8a0': [], '0ffd83bf': ['05b93cfe']})
        self.assertEqual(heads, {'A': '05b93cfe', 'B': '89bbb8a0'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ 4a60cce4 params=∅
        │        │  └─ 0ffd83bf params=∅
        │        │     └─ ● A params=∅
        │        └─ ce6cdcbc params=∅
        │           └─ ● B params=∅
        └─ 20c9a09f params=∅
        """

        # verify checksums for A
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "FILE3", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1 FILE3\n", msg=err)

        # verify A checksums
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(out, md5sum_A1)

        # create random file in A and checksum it
        rc, out, err = self.run_cmd(f"dd if=/dev/random of={self.tmpdir}/active/A/FILE4 bs=1M count=128")
        self.assertEqual(rc, 0, msg=err)
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum FILE4 >> md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "FILE3", "FILE4", "md5sum.txt"])

        # get checksums from A
        rc, md5sum_A2, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt")
        self.assertEqual(rc, 0, msg=err)

        # create head C from a parent A state
        rc, out, err = self.run_sprout("create --head C --from-run 0ffd83bf")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': ['0ffd83bf'], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': ['89bbb8a0'], '05b93cfe': [], '89bbb8a0': [], '0ffd83bf': ['05b93cfe', 'abc5beca'], 'abc5beca': []})
        self.assertEqual(heads, {'A': '05b93cfe', 'B': '89bbb8a0', 'C': 'abc5beca'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ 4a60cce4 params=∅
        │        │  └─ 0ffd83bf params=∅
        │        │     ├─ ● A params=∅
        │        │     └─ ● C params=∅
        │        └─ ce6cdcbc params=∅
        │           └─ ● B params=∅
        └─ 20c9a09f params=∅
        """

        # verify checksums for C
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/C && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/C")), ["FILE1", "FILE3", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/C && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1 FILE3\n", msg=err)

        # verify C checksums - should be equal to previous A state
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/C && cat md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(out, md5sum_A1)

        # remove head C
        rc, out, err = self.run_sprout("remove --head C")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], '4a60cce4': ['0ffd83bf'], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': ['89bbb8a0'], '05b93cfe': [], '89bbb8a0': [], '0ffd83bf': ['05b93cfe']})
        self.assertEqual(heads, {'A': '05b93cfe', 'B': '89bbb8a0'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ 4a60cce4 params=∅
        │        │  └─ 0ffd83bf params=∅
        │        │     └─ ● A params=∅
        │        └─ ce6cdcbc params=∅
        │           └─ ● B params=∅
        └─ 20c9a09f params=∅
        """

        # verify checksums for A
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && md5sum -c md5sum.txt")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(sorted(os.listdir(f"{self.tmpdir}/active/A")), ["FILE1", "FILE3", "FILE4", "md5sum.txt"])
        rc, out, err = self.run_cmd(f"cd {self.tmpdir}/active/A && cat md5sum.txt | awk '{{print $2}}' | sort | xargs")
        self.assertEqual(out, "FILE1 FILE3 FILE4\n", msg=err)

        # remove node, reparent children
        rc, out, err = self.run_sprout("remove --run 4a60cce4")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': ['fc27ffe9'], 'fc27ffe9': ['ce6cdcbc', '0ffd83bf'], 'ce6cdcbc': ['89bbb8a0'], '05b93cfe': [], '89bbb8a0': [], '0ffd83bf': ['05b93cfe']})
        self.assertEqual(heads, {'A': '05b93cfe', 'B': '89bbb8a0'})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        │     └─ fc27ffe9 params=∅
        │        ├─ ce6cdcbc params=∅
        │        │  └─ ● B params=∅
        │        └─ 0ffd83bf params=∅
        │           └─ ● A params=∅
        └─ 20c9a09f params=∅
        """

        # remove the whole branch
        rc, out, err = self.run_sprout("remove --run fc27ffe9 --whole-branch")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': [], '5ed00be8': [], '49049a55': []})
        self.assertEqual(heads, {})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        └─ 20c9a09f params=∅
        """

        # create headless run from another run
        rc, out, err = self.run_sprout("create --from-run 20c9a09f")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': ['5ed00be8', '49049a55'], '20c9a09f': ['4a60cce4'], '5ed00be8': [], '49049a55': [], '4a60cce4': []})
        self.assertEqual(heads, {})

        """
        Group: GroupA
        ├─ 2ca5a9eb params=∅
        │  ├─ 5ed00be8 params=∅
        │  └─ 49049a55 params=∅
        └─ 20c9a09f params=∅
           └─ 4a60cce4 params=∅
        """

        # remove the whole group
        rc, out, err = self.run_sprout("remove --group GroupA")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {})
        self.assertEqual(heads, {})

        if False:
            rc, out, err = self.run_sprout("tree")
            print(out)
            runs, heads, tree = self.sprout.get_tree()
            print(tree)
            print()
            print(heads)

    def test_create_with_params(self):
        # create another run with a head and empty params, description, alias
        rc, out, err = self.run_sprout("create GroupA --head A")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': []})
        self.assertEqual(heads, {'A': '2ca5a9eb'})
        self.assertEqual(runs['2ca5a9eb']['params'], {})
        self.assertEqual(runs['2ca5a9eb']['description'], "")
        self.assertEqual(runs['2ca5a9eb']['alias'], "")

        # move head A forward, but update params, description, alias
        rc, out, err = self.run_sprout("create --from-head A --params 'a1 1' --alias a1-alias --description a1-desc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['2ca5a9eb']})
        self.assertEqual(heads, {'A': '2ca5a9eb'})
        # previous head
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # updated current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a1': 1})
        self.assertEqual(runs['2ca5a9eb']['description'], "a1-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a1-alias")

        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ ● A (a-alias) params=a
        """

        # move head A forward, but update params, description, alias
        rc, out, err = self.run_sprout("create --from-head A --params 'a2 2' --alias a2-alias --description a2-desc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['5ed00be8'], '5ed00be8': ['2ca5a9eb']})
        self.assertEqual(heads, {'A': '2ca5a9eb'})
        # previous previos head
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # previous head
        self.assertEqual(runs['5ed00be8']['params'], {'a1': 1})
        self.assertEqual(runs['5ed00be8']['description'], "a1-desc")
        self.assertEqual(runs['5ed00be8']['alias'], "a1-alias")
        # updated current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a2': 2})
        self.assertEqual(runs['2ca5a9eb']['description'], "a2-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a2-alias")

        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ 5ed00be8 (a1-alias) params=a1
              └─ ● A (a2-alias) params=a2
        """

        # create head B from head A
        rc, out, err = self.run_sprout("create --head B --from-head A --params 'b1 1' --alias b1-alias --description b1-desc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['5ed00be8'], '5ed00be8': ['49049a55'], '49049a55': ['2ca5a9eb', '4a60cce4'], '4a60cce4': []})
        self.assertEqual(heads, {'A': '2ca5a9eb', 'B': '4a60cce4'})
        # previous previous previos head A
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # previous previous head A
        self.assertEqual(runs['5ed00be8']['params'], {'a1': 1})
        self.assertEqual(runs['5ed00be8']['description'], "a1-desc")
        self.assertEqual(runs['5ed00be8']['alias'], "a1-alias")
        # previous head A
        self.assertEqual(runs['49049a55']['params'], {'a2': 2})
        self.assertEqual(runs['49049a55']['description'], "a2-desc")
        self.assertEqual(runs['49049a55']['alias'], "a2-alias")
        # current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a2': 2})
        self.assertEqual(runs['2ca5a9eb']['description'], "a2-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a2-alias")
        # current head B
        self.assertEqual(runs['4a60cce4']['params'], {'b1': 1})
        self.assertEqual(runs['4a60cce4']['description'], "b1-desc")
        self.assertEqual(runs['4a60cce4']['alias'], "b1-alias")


        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ 5ed00be8 (a1-alias) params=a1
              └─ 49049a55 (a2-alias) params=a2
                 ├─ ● A (a2-alias) params=a2
                 └─ ● B (b1-alias) params=b1
        """

        # create headless snapshot from head B without params
        rc, out, err = self.run_sprout("create --from-head B")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['5ed00be8'], '5ed00be8': ['49049a55'], '49049a55': ['2ca5a9eb', 'fc27ffe9'], '4a60cce4': [], 'fc27ffe9': ['4a60cce4']})
        self.assertEqual(heads, {'A': '2ca5a9eb', 'B': '4a60cce4'})
        # previous previous previos head A
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # previous previous head A
        self.assertEqual(runs['5ed00be8']['params'], {'a1': 1})
        self.assertEqual(runs['5ed00be8']['description'], "a1-desc")
        self.assertEqual(runs['5ed00be8']['alias'], "a1-alias")
        # previous head A
        self.assertEqual(runs['49049a55']['params'], {'a2': 2})
        self.assertEqual(runs['49049a55']['description'], "a2-desc")
        self.assertEqual(runs['49049a55']['alias'], "a2-alias")
        # current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a2': 2})
        self.assertEqual(runs['2ca5a9eb']['description'], "a2-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a2-alias")
        # previous head B
        self.assertEqual(runs['fc27ffe9']['params'], {'b1': 1})
        self.assertEqual(runs['fc27ffe9']['description'], "b1-desc")
        self.assertEqual(runs['fc27ffe9']['alias'], "b1-alias")
        # current head B
        self.assertEqual(runs['4a60cce4']['params'], {'b1': 1})
        self.assertEqual(runs['4a60cce4']['description'], "b1-desc")
        self.assertEqual(runs['4a60cce4']['alias'], "b1-alias")

        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ 5ed00be8 (a1-alias) params=a1
              └─ 49049a55 (a2-alias) params=a2
                 ├─ ● A (a2-alias) params=a2
                 └─ fc27ffe9 (b1-alias) params=b1
                    └─ ● B (b1-alias) params=b1
        """

        # create headless snapshot from another run with params
        rc, out, err = self.run_sprout("create --from-run fc27ffe9 --params 'other 1' --alias other-alias --description other-desc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['5ed00be8'], '5ed00be8': ['49049a55'], '49049a55': ['2ca5a9eb', 'fc27ffe9'], '4a60cce4': [], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': []})
        self.assertEqual(heads, {'A': '2ca5a9eb', 'B': '4a60cce4'})
        # previous previous previos head A
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # previous previous head A
        self.assertEqual(runs['5ed00be8']['params'], {'a1': 1})
        self.assertEqual(runs['5ed00be8']['description'], "a1-desc")
        self.assertEqual(runs['5ed00be8']['alias'], "a1-alias")
        # previous head A
        self.assertEqual(runs['49049a55']['params'], {'a2': 2})
        self.assertEqual(runs['49049a55']['description'], "a2-desc")
        self.assertEqual(runs['49049a55']['alias'], "a2-alias")
        # current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a2': 2})
        self.assertEqual(runs['2ca5a9eb']['description'], "a2-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a2-alias")
        # previous head B
        self.assertEqual(runs['fc27ffe9']['params'], {'b1': 1})
        self.assertEqual(runs['fc27ffe9']['description'], "b1-desc")
        self.assertEqual(runs['fc27ffe9']['alias'], "b1-alias")
        # current head B
        self.assertEqual(runs['4a60cce4']['params'], {'b1': 1})
        self.assertEqual(runs['4a60cce4']['description'], "b1-desc")
        self.assertEqual(runs['4a60cce4']['alias'], "b1-alias")
        # new state
        self.assertEqual(runs['ce6cdcbc']['params'], {'other': 1})
        self.assertEqual(runs['ce6cdcbc']['description'], "other-desc")
        self.assertEqual(runs['ce6cdcbc']['alias'], "other-alias")

        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ 5ed00be8 (a1-alias) params=a1
              └─ 49049a55 (a2-alias) params=a2
                 ├─ ● A (a2-alias) params=a2
                 └─ fc27ffe9 (b1-alias) params=b1
                    ├─ ● B (b1-alias) params=b1
                    └─ ce6cdcbc (other-alias) params=other
        """

        # create headless snapshot from a run without params expecting inheritance
        rc, out, err = self.run_sprout("create --from-run ce6cdcbc")
        self.assertEqual(rc, 0, msg=err)

        runs, heads, tree = self.sprout.get_tree()
        self.assertEqual(tree, {'2ca5a9eb': [], '20c9a09f': ['5ed00be8'], '5ed00be8': ['49049a55'], '49049a55': ['2ca5a9eb', 'fc27ffe9'], '4a60cce4': [], 'fc27ffe9': ['4a60cce4', 'ce6cdcbc'], 'ce6cdcbc': ['05b93cfe'], '05b93cfe': []})
        self.assertEqual(heads, {'A': '2ca5a9eb', 'B': '4a60cce4'})
        # previous previous previos head A
        self.assertEqual(runs['20c9a09f']['params'], {})
        self.assertEqual(runs['20c9a09f']['description'], "")
        self.assertEqual(runs['20c9a09f']['alias'], "")
        # previous previous head A
        self.assertEqual(runs['5ed00be8']['params'], {'a1': 1})
        self.assertEqual(runs['5ed00be8']['description'], "a1-desc")
        self.assertEqual(runs['5ed00be8']['alias'], "a1-alias")
        # previous head A
        self.assertEqual(runs['49049a55']['params'], {'a2': 2})
        self.assertEqual(runs['49049a55']['description'], "a2-desc")
        self.assertEqual(runs['49049a55']['alias'], "a2-alias")
        # current head A
        self.assertEqual(runs['2ca5a9eb']['params'], {'a2': 2})
        self.assertEqual(runs['2ca5a9eb']['description'], "a2-desc")
        self.assertEqual(runs['2ca5a9eb']['alias'], "a2-alias")
        # previous head B
        self.assertEqual(runs['fc27ffe9']['params'], {'b1': 1})
        self.assertEqual(runs['fc27ffe9']['description'], "b1-desc")
        self.assertEqual(runs['fc27ffe9']['alias'], "b1-alias")
        # current head B
        self.assertEqual(runs['4a60cce4']['params'], {'b1': 1})
        self.assertEqual(runs['4a60cce4']['description'], "b1-desc")
        self.assertEqual(runs['4a60cce4']['alias'], "b1-alias")
        # ce6cdcbc state
        self.assertEqual(runs['ce6cdcbc']['params'], {'other': 1})
        self.assertEqual(runs['ce6cdcbc']['description'], "other-desc")
        self.assertEqual(runs['ce6cdcbc']['alias'], "other-alias")
        # new state, expect params inherited from ce6cdcbc
        self.assertEqual(runs['05b93cfe']['params'], {'other': 1})
        self.assertEqual(runs['05b93cfe']['description'], "other-desc")
        self.assertEqual(runs['05b93cfe']['alias'], "other-alias")

        """
        Group: GroupA
        └─ 20c9a09f params=∅
           └─ 5ed00be8 (a1-alias) params=a1
              └─ 49049a55 (a2-alias) params=a2
                 ├─ ● A (a2-alias) params=a2
                 └─ fc27ffe9 (b1-alias) params=b1
                    ├─ ● B (b1-alias) params=b1
                    └─ ce6cdcbc (other-alias) params=other
                       └─ 05b93cfe (other-alias) params=other
        """

        if False:
            rc, out, err = self.run_sprout("tree")
            print(out)
            runs, heads, tree = self.sprout.get_tree()
            print(tree)
            print()
            print(heads)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=sys.argv[:1] + sys.argv[2:])
    else:
        sys.exit(main())
