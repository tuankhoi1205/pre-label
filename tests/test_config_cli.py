from pathlib import Path
import pytest
import yaml
from prelabel.cli import main
from prelabel.config import load_config


@pytest.fixture
def config_file(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "mini.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["project_root"] = str(tmp_path)
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_env_config_and_relative_paths(config_file, monkeypatch):
    monkeypatch.setenv("NUSCENES_ROOT", "/example/data")
    config = load_config(config_file)
    assert Path(config["run_dir"]).is_absolute()
    assert config["dataset"]["root"].endswith("data")
    assert config["detector"]["sweeps"] == 9


@pytest.mark.parametrize("setting", ["threshold", "roi", "unknown_class", "sweeps"])
def test_invalid_config_rejected(config_file, setting):
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if setting == "threshold":
        cfg["detector"]["confidence"] = 1.1
    elif setting == "roi":
        cfg["evaluation"]["point_cloud_range"] = [1, 1, 1, -1, -1, -1]
    elif setting == "unknown_class":
        cfg["detector"]["class_thresholds"]["unknown"] = 0.3
    else:
        cfg["detector"]["sweeps"] = 0
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_file)


def test_cli_error_is_actionable_and_nonzero(config_file, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config_file), "export"])
    assert error.value.code == 2
    assert "ERROR:" in capsys.readouterr().err
