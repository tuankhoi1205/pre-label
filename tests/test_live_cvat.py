import os
import pytest
from prelabel.cvat import CVATClient, export_annotations, upload


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("PRELABEL_LIVE_CVAT") != "1", reason="No explicitly enabled live CVAT server")
def test_live_pcd_cuboid_roundtrip(run_fixture):
    """Real REST integration on synthetic PCD/boxes; this is not model inference."""
    cfg, _, boxes = run_fixture
    cfg["cvat"]["timeout_seconds"] = 600
    cfg["cvat"]["poll_seconds"] = 2
    client = CVATClient(cfg["cvat"])
    state = None
    try:
        state = upload(cfg, client)
        reviewed = export_annotations(cfg, client)
        assert len(reviewed) == len(boxes)
        assert {b.sample_token for b in reviewed} == {b.sample_token for b in boxes}
        upload(cfg, client)
        assert len(export_annotations(cfg, client)) == len(boxes)
        print(f"Synthetic geometry test task: {state['task_url']}")
    finally:
        if state and os.environ.get("CVAT_KEEP_TEST_TASK") != "1":
            client.request("DELETE", f"/api/tasks/{state['task_id']}")
