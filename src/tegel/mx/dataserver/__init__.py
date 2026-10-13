"""NSLS-II MX dataset resource type, extractor, and reader."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tegel.core.contracts.resources import DataDescriptor, Resource, StructureFamily
from tegel.core.contracts.types import ResourceType
from tegel.core.plugins import Extractor, ProcessedDataPackage

SCHEMA_ID = "https://w3id.org/tegel/mx/dataset"

RECORD = ResourceType(
    name="mx.dataset",
    version="1",
    metadata_schema=SCHEMA_ID,
)

_TECHNIQUES = {
    "cryo-ET",
    "cryo-EM",
    "RX",
    "SX",
    "SFX",
    "SAXS",
    "WAXS",
    "SANS",
    "WANS",
    "XRD",
    "EMD",
    "SEDX",
    "other",
}
_OPTIONAL = (
    "pid",
    "size",
    "wavelength_angstrom",
    "detector",
    "proposal",
    "kind",
    "space_group",
    "resolution",
    "completeness",
    "cc_half",
    "key",
    "relations",
)
_ZULU = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class MxExtractor(Extractor):
    """Claim an NSLS-II MX ``collection.json`` and point at its frame store."""

    @property
    def plugin_id(self) -> str:
        return "mx-dataset"

    def validate(self, input_path: str) -> bool:
        path = Path(input_path)
        if path.name != "collection.json" or not path.is_file():
            return False
        body = _object(path)
        if body is None:
            return False
        return (
            _facility_token(body.get("facility")) == "NSLSII"
            and _protein(body) is not None
            and _instrument(body) is not None
        )

    def process(self, input_path: str, options: dict[str, Any]) -> ProcessedDataPackage:
        path = Path(input_path)
        body = _object(path)
        if body is None:
            raise ValueError("MX collection is not a JSON object")
        metadata = _catalog_metadata(body)
        metadata["plugin_id"] = self.plugin_id
        metadata["source_name"] = path.name
        metadata["payload_uri"] = _payload_uri(body, path)
        metadata["resource_type"] = RECORD.name
        metadata["resource_type_version"] = RECORD.version
        caller_key = options.get("key")
        if isinstance(caller_key, str) and caller_key.strip() and caller_key == caller_key.strip():
            metadata["key"] = caller_key
        return ProcessedDataPackage(payload_path=str(path), metadata=metadata, metrics={})


class MxReader:
    """Describe a diffraction array from ``zarr.json``. Chunk files are not opened."""

    @property
    def plugin_id(self) -> str:
        return "mx-dataset"

    def validate(self, resource: Resource) -> bool:
        return resource.type == RECORD.name

    def describe(self, resource: Resource) -> DataDescriptor:
        uri = resource.metadata.get("payload_uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("payload_uri is required")
        array = _array_metadata(_local_path(uri))
        if array is None:
            return DataDescriptor(
                family=StructureFamily.bytes,
                media_type="application/octet-stream",
                content_url=f"{resource.uri}/data/content",
            )
        shape = [int(item) for item in array["shape"]]
        return DataDescriptor(
            family=StructureFamily.array,
            media_type="application/zarr",
            shape=shape,
            chunks=[_chunk_shape(array, shape)],
            data_type=str(array["data_type"]),
        )


def _catalog_metadata(body: dict[str, Any]) -> dict[str, Any]:
    """Map a collection or lambda record onto the facility search fields."""
    experiment_id = (
        _text(body.get("experiment_id"))
        or _text(body.get("uuid"))
        or _text(body.get("dataset_uuid"))
    )
    metadata: dict[str, Any] = {
        "facility": "NSLSII",
        "technique": _technique(body),
    }
    protein = _protein(body)
    instrument = _instrument(body)
    if protein:
        metadata["protein_name"] = protein
    if instrument:
        metadata["instrument"] = instrument
    if experiment_id:
        metadata["experiment_id"] = experiment_id
        metadata["uuid"] = experiment_id
    created = _timestamp(body.get("creation_date") or body.get("collected"))
    if created:
        metadata["creation_date"] = created
    public = _public_flag(body.get("is_public"))
    if public:
        metadata["is_public"] = public
    seguid = _seguid(body.get("seguid"))
    if seguid:
        metadata["seguid"] = seguid
    for field in _OPTIONAL:
        if field == "size":
            raw_size = body.get("size")
            if raw_size in (None, ""):
                raw_size = body.get("size_bytes")
            text = _text(raw_size)
        elif field == "relations":
            text = body.get(field)
        else:
            text = _text(body.get(field))
        if text not in (None, ""):
            metadata[field] = text
    return metadata


def _protein(body: dict[str, Any]) -> str | None:
    return (
        _text(body.get("protein_name"))
        or _text(body.get("sample_name"))
        or _text(body.get("sample_code"))
    )


def _instrument(body: dict[str, Any]) -> str | None:
    return (
        _text(body.get("instrument"))
        or _text(body.get("beamline"))
        or _text(body.get("instrument_code"))
    )


def _facility_token(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    token = text.upper().replace("-", "").replace("_", "").replace(" ", "")
    if token in {"NSLSII", "NSLS2"}:
        return "NSLSII"
    return None


def _technique(body: dict[str, Any]) -> str:
    raw = _text(body.get("technique")) or _text(body.get("technique_lambda"))
    if raw is None:
        return "RX"
    if raw in _TECHNIQUES:
        return raw
    token = raw.upper().replace(" ", "").replace("-", "_")
    if token in {"MX", "XRAY_CRYSTALLOGRAPHY"}:
        return "RX"
    return "RX"


def _timestamp(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    if _ZULU.match(text):
        return text
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
        return f"{text}T00:00:00Z"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", text):
        return f"{text}Z"
    return None


def _public_flag(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered
    return None


def _seguid(value: object) -> str | None:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return ",".join(parts) or None
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        found = [part for part in parts if part]
        return ",".join(found) or None
    return None


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return format(value, "g")
    return None


def _object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    return body


def _payload_uri(body: dict[str, Any], record: Path) -> str:
    explicit = body.get("payload_uri")
    if isinstance(explicit, str) and explicit.strip():
        if explicit.startswith("file:"):
            return explicit
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = record.parent / candidate
        return candidate.resolve().as_uri()
    return record.resolve().as_uri()


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _array_metadata(path: Path) -> dict[str, Any] | None:
    document = path / "zarr.json" if path.is_dir() else path
    if document.name != "zarr.json" or not document.is_file():
        return None
    try:
        body = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict) or body.get("node_type") != "array":
        return None
    shape = body.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    if not isinstance(body.get("data_type"), str) or not body["data_type"]:
        return None
    return body


def _chunk_shape(array: dict[str, Any], shape: list[int]) -> list[int]:
    grid = array.get("chunk_grid")
    if isinstance(grid, dict):
        configuration = grid.get("configuration")
        if isinstance(configuration, dict):
            chunk = configuration.get("chunk_shape")
            if isinstance(chunk, list) and chunk:
                return [int(item) for item in chunk]
    return shape
