"""Stage NSLS-II MX collections into a Tegel catalog.

Discovery follows the facility layout used by
https://github.com/phzwart/etldatadash : one record per crystal pin whose name
starts with ``st_`` or ``MB-VW-`` and that sits under an ``mx*`` visit folder.
Frame, MTZ, and HDF5 bytes stay where they are. This writes a ``collection.json``
manifest and a staging catalog.

``ingest`` reads an existing DATA_ROOT of ``lambda_mx_record.json`` files from
that project and projects them into the same manifest.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tegel.mx.dataserver import _public_flag, _seguid, _technique, _timestamp

UUID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
CRYSTAL_NAME_RE = re.compile(r"^(MB-VW-|st_)", re.IGNORECASE)
RUN_DIR_RE = re.compile(r"^\d+$")
PASS_DIR_RE = re.compile(r"^pass-\d+", re.IGNORECASE)
DEFAULT_BEAMLINES = ("fmx", "amx", "nyx")
PROCESSING_DIRS = ("autoProcOutput", "fastDPOutput", "dimpleOutput", "dimple_autoProc_dir")
QUALITY_FIELDS = (
    "wavelength_angstrom",
    "space_group",
    "resolution",
    "completeness",
    "cc_half",
    "detector",
)


class MxEtlError(Exception):
    """The facility tree or a lambda record could not be staged."""


@dataclass(frozen=True)
class CrystalCollection:
    """One crystal pin. Payloads remain on ``path``."""

    path: Path
    sample_code: str
    dataset_uuid: str
    kind: str
    beamline: str | None
    visit: Path


def dataset_uuid_for(path: Path) -> str:
    """Match the lambda-rocrate id for this pin directory."""
    resolved = str(path.expanduser().resolve())
    return str(uuid.uuid5(UUID_NAMESPACE, f"lambda-rocrate:fmx:{resolved}"))


def beamline_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for beamline in DEFAULT_BEAMLINES:
        if beamline in parts:
            return beamline
    return None


def discover_collections_from_roots(
    roots: Sequence[str | Path],
    *,
    beamlines: tuple[str, ...] | None = None,
    include_raster_only: bool = True,
    include_empty: bool = True,
) -> tuple[list[Path], list[CrystalCollection]]:
    """Expand visit, pass, or beamline-glob roots into one collection per pin."""
    expanded = _expand_roots(roots, beamlines or DEFAULT_BEAMLINES)
    visits: list[Path] = []
    seen_visits: set[Path] = set()
    for root in expanded:
        for visit in _resolve_visits(root):
            if visit in seen_visits:
                continue
            seen_visits.add(visit)
            visits.append(visit)
    collections: list[CrystalCollection] = []
    seen_pins: set[Path] = set()
    for visit in visits:
        for item in _discover_visit(visit):
            if item.path in seen_pins:
                continue
            if item.kind == "raster_only" and not include_raster_only:
                continue
            if item.kind == "empty" and not include_empty:
                continue
            seen_pins.add(item.path)
            collections.append(item)
    return visits, collections


def collection_document(collection: CrystalCollection) -> dict[str, str]:
    """Manifest fields the MX extractor copies into the catalog."""
    if not collection.beamline:
        raise MxEtlError(f"{collection.sample_code}: beamline is not in the path")
    payload = _primary_master(collection.path) or collection.path
    document = {
        "protein_name": collection.sample_code,
        "experiment_id": collection.dataset_uuid,
        "creation_date": _collected(collection.path),
        "facility": "NSLSII",
        "technique": "RX",
        "instrument": collection.beamline.upper(),
        "kind": collection.kind,
        "payload_uri": payload.resolve().as_uri(),
    }
    proposal = _proposal(collection.path)
    if proposal:
        document["proposal"] = proposal
    return document


def project_lambda_record(body: dict[str, Any], crate_dir: Path) -> dict[str, str] | None:
    """Project a ``lambda_mx_record.json`` object onto a collection manifest."""
    sample = (
        _text(body.get("protein_name"))
        or _text(body.get("sample_code"))
        or _text(body.get("title"))
    )
    identifier = _text(body.get("dataset_uuid")) or _text(body.get("id"))
    beamline = (
        _text(body.get("instrument"))
        or _text(body.get("beamline"))
        or _text(body.get("instrument_code"))
    )
    if not sample or not identifier or not beamline:
        return None
    if "/" in identifier or "\\" in identifier:
        return None
    source = body.get("source_path")
    if isinstance(source, str) and source.strip():
        source_path = Path(source).expanduser()
        payload = source_path.resolve().as_uri()
        proposal = _proposal(source_path)
    else:
        payload = crate_dir.resolve().as_uri()
        proposal = _proposal(crate_dir)
    document = {
        "protein_name": _text(body.get("protein_name")) or sample,
        "experiment_id": identifier,
        "creation_date": _timestamp(body.get("creation_date")) or _collected(crate_dir),
        "facility": "NSLSII",
        "technique": _technique(body),
        "instrument": beamline.upper(),
        "payload_uri": payload,
    }
    public = _public_flag(body.get("is_public"))
    if public:
        document["is_public"] = public
    seguid = _seguid(body.get("seguid"))
    if seguid:
        document["seguid"] = seguid
    pid = _text(body.get("pid"))
    if pid:
        document["pid"] = pid
    raw_size = body.get("size")
    if raw_size in (None, ""):
        raw_size = body.get("size_bytes")
    size = _text(raw_size)
    if size:
        document["size"] = size
    kind = _text(body.get("kind"))
    if kind:
        document["kind"] = kind
    if proposal:
        document["proposal"] = proposal
    for field in QUALITY_FIELDS:
        text = _text(body.get(field))
        if text:
            document[field] = text
    return document


def write_collection(output: Path, document: dict[str, str]) -> Path:
    directory = output / document["experiment_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "collection.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def stage_records(paths: list[str], catalog: Path) -> int:
    """Write claimed manifests into a new staging catalog."""
    from tegel.actions.staging import write_staging

    catalog.parent.mkdir(parents=True, exist_ok=True)
    return write_staging(paths, catalog)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``facility-build`` or ``ingest``."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "facility-build":
            return _facility_build(args)
        return _ingest(args)
    except MxEtlError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _facility_build(args: argparse.Namespace) -> int:
    beamlines = _beamlines(args.beamlines)
    _visits, collections = discover_collections_from_roots(
        args.roots,
        beamlines=beamlines,
        include_raster_only=not args.exclude_raster_only,
        include_empty=not args.exclude_empty,
    )
    if args.limit is not None:
        collections = collections[: args.limit]
    ready = [item for item in collections if item.beamline]
    skipped = len(collections) - len(ready)
    if args.dry_run:
        for item in ready:
            print(f"{item.sample_code}\t{item.dataset_uuid}\tkind={item.kind}\t{item.beamline}")
        print(f"{len(ready)} collection(s); skipped {skipped} without a beamline")
        return 0
    if args.output is None:
        print("error: -o is required unless --dry-run", file=sys.stderr)
        return 2
    paths = [str(write_collection(args.output, collection_document(item))) for item in ready]
    for item in ready:
        print(f"{item.sample_code}\t{item.dataset_uuid}\tkind={item.kind}")
    if not paths:
        print("0 collection(s)")
        return 0
    catalog = _catalog_path(args.output, args.catalog)
    count = stage_records(paths, catalog)
    print(f"staged {count} record(s) into {catalog}")
    return 0 if count == len(paths) else 1


def _ingest(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise MxEtlError(f"path does not exist: {data_root}")
    documents: list[dict[str, str]] = []
    for path in sorted(data_root.rglob("lambda_mx_record.json")):
        if _hidden(path) or not path.is_file():
            continue
        body = _object(path)
        if body is None:
            print(f"skip {path}: not a JSON object", file=sys.stderr)
            continue
        document = project_lambda_record(body, path.parent)
        if document is None:
            print(f"skip {path}: sample, uuid, or beamline is missing", file=sys.stderr)
            continue
        documents.append(document)
        if args.limit is not None and len(documents) >= args.limit:
            break
    if args.dry_run:
        for document in documents:
            print(f"{document['sample_name']}\t{document['uuid']}\t{document['beamline']}")
        print(f"{len(documents)} record(s)")
        return 0
    if args.output is None:
        print("error: -o is required unless --dry-run", file=sys.stderr)
        return 2
    paths = [str(write_collection(args.output, document)) for document in documents]
    if not paths:
        print("0 record(s)")
        return 0
    catalog = _catalog_path(args.output, args.catalog)
    count = stage_records(paths, catalog)
    print(f"staged {count} record(s) into {catalog}")
    return 0 if count == len(paths) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tegel-mx-etl",
        description=(
            "Stage NSLS-II MX pins into a Tegel catalog. "
            "Payload files are not copied."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    facility = commands.add_parser(
        "facility-build",
        help="Discover st_* and MB-VW-* pins under mx* visits and stage them.",
    )
    facility.add_argument(
        "roots",
        nargs="+",
        help="Visit, pass-*, or a glob. __BEAMLINE__ expands.",
    )
    _output_flags(facility)
    facility.add_argument(
        "--beamlines",
        default=",".join(DEFAULT_BEAMLINES),
        help="Beamlines substituted for __BEAMLINE__.",
    )
    facility.add_argument("--exclude-empty", action="store_true")
    facility.add_argument("--exclude-raster-only", action="store_true")
    ingest = commands.add_parser(
        "ingest",
        help="Project lambda_mx_record.json files from a DATA_ROOT and stage them.",
    )
    ingest.add_argument("data_root", help="Directory of {uuid}/lambda_mx_record.json crates.")
    _output_flags(ingest)
    return parser


def _output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Directory for collection.json manifests.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Staging catalog. Defaults to OUTPUT/catalog.db.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the records and write nothing.",
    )


def _catalog_path(output: Path, catalog: Path | None) -> Path:
    return catalog if catalog is not None else output / "catalog.db"


def _beamlines(text: str) -> tuple[str, ...]:
    found = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    return found or DEFAULT_BEAMLINES


def _expand_roots(patterns: Sequence[str | Path], beamlines: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for raw in patterns:
        text = str(raw)
        if "__BEAMLINE__" in text:
            variants = [text.replace("__BEAMLINE__", name) for name in beamlines]
        else:
            variants = [text]
        for variant in variants:
            if any(mark in variant for mark in "*?[]"):
                hits = [Path(item) for item in sorted(glob.glob(variant))]
                if not hits:
                    raise MxEtlError(f"glob matched nothing: {variant}")
                matches.extend(hits)
                continue
            path = Path(variant).expanduser()
            try:
                exists = path.exists()
            except PermissionError as exc:
                raise _denied(path) from exc
            if not exists:
                raise MxEtlError(f"path does not exist: {variant}")
            matches.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in matches:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_visits(root: Path) -> list[Path]:
    try:
        root = root.expanduser().resolve()
    except PermissionError as exc:
        raise _denied(root) from exc
    if not root.is_dir():
        raise MxEtlError(f"not a directory: {root}")
    if root.name.startswith("mx"):
        parent = root.parent
        return [parent if _has_mx_child(parent) else root]
    if _has_mx_child(root):
        return [root]
    if PASS_DIR_RE.match(root.name):
        return [child.resolve() for child in _dirs(root) if _has_mx_child(child)]
    visits: list[Path] = []
    for child in _dirs(root):
        if PASS_DIR_RE.match(child.name) or _has_mx_child(child):
            if PASS_DIR_RE.match(child.name):
                visits.extend(_resolve_visits(child))
            else:
                visits.append(child.resolve())
            continue
        for grandchild in _dirs(child):
            if PASS_DIR_RE.match(grandchild.name) or _has_mx_child(grandchild):
                visits.extend(_resolve_visits(grandchild))
    seen: set[Path] = set()
    unique: list[Path] = []
    for visit in visits:
        if visit in seen:
            continue
        seen.add(visit)
        unique.append(visit)
    return unique


def _discover_visit(visit: Path) -> list[CrystalCollection]:
    beamline = beamline_from_path(visit)
    found: list[CrystalCollection] = []
    for mx_root in _mx_roots(visit):
        for child in sorted(mx_root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or not CRYSTAL_NAME_RE.match(child.name):
                continue
            kind = _classify(child)
            if kind is None:
                kind = "empty"
            found.append(
                CrystalCollection(
                    path=child.resolve(),
                    sample_code=child.name,
                    dataset_uuid=dataset_uuid_for(child),
                    kind=kind,
                    beamline=beamline,
                    visit=visit,
                )
            )
    return found


def _classify(path: Path) -> str | None:
    any_files = False
    full = False
    for run in _run_dirs(path):
        for puck in _puck_dirs(run):
            if any(child.is_file() for child in puck.rglob("*")):
                any_files = True
            if _has_master(puck) or _has_processing(puck):
                full = True
    if full:
        return "full"
    if any_files or any(child.is_file() for child in path.rglob("*")):
        return "raster_only"
    return None


def _has_master(puck: Path) -> bool:
    for path in puck.glob("*_master.h5"):
        if path.is_file() and "raster" not in path.name.lower():
            return True
    return False


def _has_processing(puck: Path) -> bool:
    return any((puck / name).is_dir() for name in PROCESSING_DIRS)


def _primary_master(collection: Path) -> Path | None:
    found: list[tuple[int, Path]] = []
    for run in _run_dirs(collection):
        for puck in _puck_dirs(run):
            for path in sorted(puck.glob("*_master.h5")):
                if path.is_file() and "raster" not in path.name.lower():
                    found.append((int(run.name), path))
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[-1][1]


def _mx_roots(visit: Path) -> list[Path]:
    roots = [child for child in _dirs(visit) if child.name.startswith("mx")]
    if roots:
        return roots
    if visit.name.startswith("mx"):
        return [visit]
    return []


def _run_dirs(collection: Path) -> list[Path]:
    runs = [child for child in _dirs(collection) if RUN_DIR_RE.match(child.name)]
    return sorted(runs, key=lambda path: int(path.name))


def _puck_dirs(run: Path) -> list[Path]:
    return [child for child in _dirs(run) if not child.name.startswith(".")]


def _dirs(path: Path) -> list[Path]:
    try:
        children = [child for child in path.iterdir() if child.is_dir()]
        return sorted(children, key=lambda item: item.name)
    except PermissionError as exc:
        raise _denied(path) from exc
    except OSError as exc:
        raise MxEtlError(f"cannot read {path}: {exc}") from exc


def _has_mx_child(path: Path) -> bool:
    try:
        return any(child.is_dir() and child.name.startswith("mx") for child in path.iterdir())
    except OSError:
        return False


def _proposal(path: Path) -> str | None:
    for part in path.parts:
        if PASS_DIR_RE.match(part):
            return part
    return None


def _collected(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return format(value, "g")
    return None


def _object(path: Path) -> dict[str, Any] | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    return body


def _hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _denied(path: Path) -> MxEtlError:
    return MxEtlError(
        f"cannot read {path}: permission denied. "
        "You must be in the Unix group for that proposal (pass-NNNNNN) on this "
        "filesystem; being listed on the PASS website is not enough."
    )


if __name__ == "__main__":
    raise SystemExit(main())
