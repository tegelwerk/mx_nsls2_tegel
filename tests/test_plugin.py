"""The MX plugin claims an NSLS-II collection and serves it read-only."""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tegel.core.contracts.resources import (
    Resource,
    StructureFamily,
    resource_id_from_metadata,
    resource_uri,
)
from tegel.core.plugins import check_package
from tegel.dataserver.app import create_app
from tegel.dataserver.catalog import insert_record
from tegel.dataserver.config import ServerConfig
from tegel.dataserver.spec import load_spec
from tegel.dataserver.testing import OIDCIssuer, assert_read_only_api
from tegel.mx.dataserver import RECORD, MxExtractor, MxReader

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "amx-lysozyme"
COLLECTION = FIXTURE / "collection.json"
PROFILE = Path(__file__).resolve().parents[1] / "schema" / "profile.yaml"
CHUNK = FIXTURE / "frames.zarr" / "c" / "0" / "0" / "0"


def test_load_spec_accepts_the_profile_and_rejects_unknown_slots(tmp_path: Path) -> None:
    profile, schema = load_spec(PROFILE)
    assert profile.id == "https://w3id.org/tegel/mx/profile"
    assert profile.schema_document == "dataset.yaml"
    assert schema["id"] == RECORD.metadata_schema
    bad = tmp_path / "profile.yaml"
    schema_path = PROFILE.parent / "dataset.yaml"
    bad.write_text(
        "\n".join(
            [
                "id: https://w3id.org/tegel/mx/profile",
                "name: MX datasets",
                f"schema: {schema_path}",
                "search:",
                "  - not_a_field",
                "properties:",
                "  - beamline",
                "entity:",
                "  name: protein_name",
                "  identifier: experiment_id",
                "  dateCreated: creation_date",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not_a_field"):
        load_spec(bad)


def test_extractor_accepts_the_fixture_and_reader_skips_chunks(tmp_path: Path) -> None:
    rejected = tmp_path / "collection.json"
    rejected.write_text(
        json.dumps({"facility": "SSRL", "sample_name": "lysozyme", "beamline": "BL12-2"}),
        encoding="utf-8",
    )
    plugin = MxExtractor()
    assert plugin.validate(str(COLLECTION)) is True
    assert plugin.validate(str(rejected)) is False
    package = check_package(plugin, plugin.process(str(COLLECTION), {"key": "amx-lysozyme-001"}))
    assert package.metadata["plugin_id"] == "mx-dataset"
    assert package.metadata["source_name"] == "collection.json"
    assert package.metadata["protein_name"] == "lysozyme"
    assert package.metadata["instrument"] == "AMX"
    assert package.metadata["facility"] == "NSLSII"
    assert package.metadata["technique"] == "RX"
    assert package.metadata["creation_date"] == "2024-06-12T00:00:00Z"
    assert package.metadata["uuid"] == "amx-lysozyme-001"
    assert package.metadata["key"] == "amx-lysozyme-001"
    assert package.metadata["resource_type"] == RECORD.name
    assert package.payload_path.endswith("collection.json")
    unkeyed = plugin.process(str(COLLECTION), {})
    assert "key" not in unkeyed.metadata
    assert resource_id_from_metadata(unkeyed.metadata, {}) == "amx-lysozyme-001"

    resource = Resource(
        id="amx-lysozyme-001",
        uri=resource_uri("http://example.test", "amx-lysozyme-001"),
        type=RECORD.name,
        metadata=package.metadata,
    )
    reader = MxReader()
    assert reader.validate(resource) is True
    assert reader.validate(resource.model_copy(update={"type": "toy.record"})) is False
    opened: list[Path] = []
    real_open = io.open

    def spy(file: object, *args: object, **kwargs: object) -> object:
        opened.append(Path(str(file)).resolve())
        return real_open(file, *args, **kwargs)

    monkeypatch_open = pytest.MonkeyPatch()
    monkeypatch_open.setattr(io, "open", spy)
    try:
        descriptor = reader.describe(resource)
    finally:
        monkeypatch_open.undo()
    assert descriptor.family is StructureFamily.array
    assert descriptor.shape == [120, 3110, 3269]
    assert descriptor.chunks == [[1, 3110, 3269]]
    assert descriptor.data_type == "uint16"
    assert CHUNK.resolve() not in opened
    assert any(path.name == "zarr.json" for path in opened)

    raw = resource.model_copy(
        update={"metadata": {**package.metadata, "payload_uri": COLLECTION.resolve().as_uri()}}
    )
    blob = reader.describe(raw)
    assert blob.family is StructureFamily.bytes
    assert blob.media_type == "application/octet-stream"


def test_read_api_rejects_writes_and_unknown_search(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.db"
    plugin = MxExtractor()
    with OIDCIssuer() as issued:
        config = ServerConfig(
            public_origin="http://test",
            catalog_uri=f"sqlite:///{catalog}",
            profile_path=PROFILE,
            issuer_url=issued.issuer,
            audience=issued.audience,
            client_id=issued.client_id,
            client_secret=issued.client_secret,
            session_secret="test-session-secret",
        )
        app = create_app(config)
        package = check_package(
            plugin,
            plugin.process(str(COLLECTION), {"key": "amx-lysozyme-001"}),
        )
        insert_record(
            app.state.tegel_catalog,
            key="amx-lysozyme-001",
            metadata=package.metadata,
            structure_family="array",
        )
        with TestClient(app) as http_client:
            assert_read_only_api(
                "http://testserver",
                token=issued.token,
                http_client=http_client,
                resource_id="amx-lysozyme-001",
                search_field="instrument",
                search_value="AMX",
            )
