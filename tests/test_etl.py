"""Facility discovery stages one pin and leaves the master file in place."""

import json
import sqlite3
from pathlib import Path

from tegel.core.plugins import check_package
from tegel.mx.dataserver import MxExtractor
from tegel.mx.etl import dataset_uuid_for, main

PASS = Path("amx") / "proposals" / "2026-2" / "pass-311234" / "visit" / "mx"


def test_facility_build_stages_the_pin_and_skips_other_names(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    puck = root / PASS / "st_lysozyme" / "1" / "A"
    puck.mkdir(parents=True)
    master = puck / "lysozyme_master.h5"
    master.write_bytes(b"master")
    other = root / PASS / "not-a-pin" / "1" / "A"
    other.mkdir(parents=True)
    (other / "ignored_master.h5").write_bytes(b"no")
    raster = root / PASS / "st_raster" / "2" / "B"
    raster.mkdir(parents=True)
    (raster / "frame.cbf").write_bytes(b"cbf")
    output = tmp_path / "records"
    code = main(
        [
            "facility-build",
            str(root / "amx" / "proposals" / "2026-2" / "pass-311234"),
            "--exclude-empty",
            "--exclude-raster-only",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    pin = root / PASS / "st_lysozyme"
    identifier = dataset_uuid_for(pin)
    manifest = output / identifier / "collection.json"
    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["protein_name"] == "st_lysozyme"
    assert body["experiment_id"] == identifier
    assert body["instrument"] == "AMX"
    assert body["facility"] == "NSLSII"
    assert body["technique"] == "RX"
    assert body["creation_date"].endswith("Z")
    assert body["proposal"] == "pass-311234"
    assert body["kind"] == "full"
    assert body["payload_uri"] == master.resolve().as_uri()
    assert master.read_bytes() == b"master"
    plugin = MxExtractor()
    package = check_package(plugin, plugin.process(str(manifest), {}))
    assert package.metadata["uuid"] == identifier
    assert package.metadata["kind"] == "full"
    with sqlite3.connect(output / "catalog.db") as connection:
        row = connection.execute("SELECT key FROM nodes WHERE key = ?", (identifier,)).fetchone()
    assert row == (identifier,)
    assert not (output / dataset_uuid_for(root / PASS / "not-a-pin")).exists()
    assert not (output / dataset_uuid_for(root / PASS / "st_raster")).exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    puck = root / PASS / "st_lysozyme" / "1" / "A"
    puck.mkdir(parents=True)
    (puck / "lysozyme_master.h5").write_bytes(b"master")
    output = tmp_path / "records"
    code = main(
        [
            "facility-build",
            str(root / "amx"),
            "--dry-run",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    assert not output.exists()


def test_ingest_projects_a_lambda_record(tmp_path: Path) -> None:
    source = tmp_path / "pin"
    source.mkdir()
    crate = tmp_path / "crates" / "amx-lysozyme-001"
    crate.mkdir(parents=True)
    (crate / "lambda_mx_record.json").write_text(
        json.dumps(
            {
                "dataset_uuid": "amx-lysozyme-001",
                "sample_code": "st_lysozyme",
                "beamline": "AMX",
                "facility": "NSLS-II",
                "creation_date": "2024-06-12T15:04:00Z",
                "source_path": str(source),
                "wavelength_angstrom": 0.9201,
                "space_group": "P 21 21 21",
                "resolution": 1.45,
                "completeness": 98.2,
                "cc_half": 0.998,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "records"
    code = main(["ingest", str(tmp_path / "crates"), "-o", str(output)])
    assert code == 0
    body = json.loads((output / "amx-lysozyme-001" / "collection.json").read_text(encoding="utf-8"))
    assert body["protein_name"] == "st_lysozyme"
    assert body["facility"] == "NSLSII"
    assert body["technique"] == "RX"
    assert body["instrument"] == "AMX"
    assert body["creation_date"] == "2024-06-12T15:04:00Z"
    assert body["wavelength_angstrom"] == "0.9201"
    assert body["space_group"] == "P 21 21 21"
    assert body["resolution"] == "1.45"
    assert body["payload_uri"] == source.resolve().as_uri()
    plugin = MxExtractor()
    manifest = output / "amx-lysozyme-001" / "collection.json"
    package = check_package(plugin, plugin.process(str(manifest), {}))
    assert package.metadata["space_group"] == "P 21 21 21"
    assert package.metadata["cc_half"] == "0.998"
    with sqlite3.connect(output / "catalog.db") as connection:
        row = connection.execute(
            "SELECT key FROM nodes WHERE key = ?",
            ("amx-lysozyme-001",),
        ).fetchone()
    assert row == ("amx-lysozyme-001",)
