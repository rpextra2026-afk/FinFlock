import pytest


def test_market_dir_is_found_in_a_folder_unpacked_from_a_left_over_zip_and_a_missing_one_is_listed(tmp_path):
    import zipfile

    from oryxis.compute.session import SessionError
    from oryxis.stage.run_stage import find_market_dir

    empty = tmp_path / "empty"
    (empty / "other").mkdir(parents=True)
    (empty / "other" / "persona-bios.bundle").write_bytes(b"x")
    with pytest.raises(SessionError, match="persona-bios.bundle"):
        find_market_dir(empty)
    zipped = tmp_path / "zipped" / "data"
    zipped.mkdir(parents=True)
    with zipfile.ZipFile(zipped / "pack.zip", "w") as bundle:
        bundle.writestr("oryxis-upload-tessera/aliases.parquet", b"x")
        bundle.writestr("oryxis-upload-tessera/prestaged/personas-tessera/units.parquet", b"y")
    found = find_market_dir(tmp_path / "zipped")
    assert (found / "aliases.parquet").exists() and (found / "prestaged" / "personas-tessera" / "units.parquet").exists()
    direct = tmp_path / "direct" / "ds" / "folder"
    direct.mkdir(parents=True)
    (direct / "aliases.parquet").write_bytes(b"x")
    assert find_market_dir(tmp_path / "direct") == direct


def test_market_dir_is_found_through_a_symlinked_dataset_folder(tmp_path):
    from oryxis.stage.run_stage import find_market_dir

    real = tmp_path / "real" / "folder"
    real.mkdir(parents=True)
    (real / "aliases.parquet").write_bytes(b"x")
    mount = tmp_path / "input"
    mount.mkdir()
    (mount / "dataset").symlink_to(tmp_path / "real", target_is_directory=True)
    assert (find_market_dir(mount) / "aliases.parquet").exists()


def test_market_zip_is_found_inside_a_symlinked_dataset_folder(tmp_path):
    import zipfile

    from oryxis.stage.run_stage import find_market_dir

    real = tmp_path / "real"
    real.mkdir()
    with zipfile.ZipFile(real / "upload.zip", "w") as z:
        z.writestr("oryxis-upload-tessera/aliases.parquet", b"x")
    root = tmp_path / "input" / "datasets" / "owner"
    root.mkdir(parents=True)
    (root / "slug").symlink_to(real, target_is_directory=True)
    assert (find_market_dir(tmp_path / "input") / "aliases.parquet").exists()
