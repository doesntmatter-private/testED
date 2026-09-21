from autodiag.obd.dtc_db import describe


def test_simulated_reader_parses_fixture(misfire_snapshot):
    s = misfire_snapshot
    assert s.vehicle.make == "Honda"
    codes = {d.code: d for d in s.dtcs}
    assert codes["P0301"].status == "stored"
    assert codes["P0300"].status == "pending"
    assert "Cylinder 1" in codes["P0301"].description  # filled from dtc_db
    assert s.pid("RPM").value == 715.0
    assert s.pid("RPM", "freeze_frame").value == 742.0
    assert s.readiness["MIL"] == "on"
    assert s.source.startswith("simulated:")


def test_redacted_removes_vin(misfire_snapshot):
    misfire_snapshot.vehicle.vin = "1HGCR2F5XEA000000"
    r = misfire_snapshot.redacted()
    assert r.vehicle.vin is None
    assert misfire_snapshot.vehicle.vin is not None


def test_describe_unknown_code():
    assert "Unknown" in describe("P9999")
    assert describe("p0301") == describe("P0301")
