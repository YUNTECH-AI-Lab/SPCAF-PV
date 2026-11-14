import time

from synpv.webapp import create_app


def test_run_through_api(tmp_path):
    app = create_app(str(tmp_path / "runs"))
    c = app.test_client()
    d = c.get("/api/defaults").get_json()
    assert "config" in d and d["steps"]
    assert c.get("/").status_code == 200

    bad = c.post("/api/run", json=dict(LATITUDE=40.7, LONGITUDE=-73.5, N_HOUSES=0))
    assert bad.status_code == 400

    r = c.post("/api/run", json=dict(LATITUDE=40.7259, LONGITUDE=-73.5143, START_DATE="2023-06-01",
                                     END_DATE="2023-06-03", N_HOUSES=10, SITE_NAME="Test site"))
    assert r.status_code == 200, r.get_json()
    job = r.get_json()["id"]
    assert c.post("/api/run", json=dict(LATITUDE=0, LONGITUDE=0)).status_code == 409   # one run at a time
    for _ in range(600):
        s = c.get(f"/api/jobs/{job}").get_json()
        if s["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    assert s["status"] == "done", s.get("error")
    assert s["metrics"]["n_houses"] == 10
    assert any(f.endswith(".png") for f in s["files"])
    z = c.get(f"/api/jobs/{job}/download.zip")
    assert z.status_code == 200 and z.data[:2] == b"PK"
    assert c.get(f"/api/jobs/{job}/files/config_used.json").status_code == 200
    assert c.get("/api/geocode?q=Munich").get_json()[0]["lat"] > 48
