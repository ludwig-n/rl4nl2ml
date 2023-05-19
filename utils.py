import pathlib
import runpy
import shutil

import kaggle
import kaggle.rest
import nbconvert
import nbformat


def download_kaggle_files(
    comp_id: str,
    dest_dir: str | pathlib.Path | None = None,
    unzip_depth: int = 2,
    verbose: bool = True
) -> None:
    """
    Downloads all files from a Kaggle competition.

    :param comp_id: Kaggle competition name (like `cornell-birdcall-identification`)
    :param dest_dir: folder to download files to, defaults to `{comp_id}/data`
    :param unzip_depth: depth to unzip archives inside archives
    :param verbose: if True, print progress information to stdout
    """
    dest_dir = pathlib.Path(dest_dir if dest_dir is not None else f'{comp_id}/data')
    api = kaggle.KaggleApi()
    api.authenticate()

    if verbose:
        print(f'Downloading files from competition {comp_id}...')
    try:
        api.competition_download_files(comp_id, path=dest_dir, quiet=not verbose)
    except kaggle.rest.ApiException as e:
        if e.status in [401, 403]:
            print(f'Permission denied. You probably need to accept the rules: https://www.kaggle.com/c/{comp_id}/rules')
            return
        else:
            raise

    for _ in range(unzip_depth):
        zips = list(dest_dir.rglob('*.zip'))
        if not zips:
            break
        for zip_path in zips:
            if verbose:
                print(f'Unpacking {zip_path}...')
            shutil.unpack_archive(zip_path, zip_path.parent)
            zip_path.unlink()

    if verbose:
        print(f'Downloaded files from competition {comp_id}')


def run_solution(path: str | pathlib.Path) -> None:
    """
    Executes a Jupyter notebook (if the file extension is .ipynb) or Python script.
    Warning: this function can execute arbitrary code!

    :param path: path to script or notebook
    """
    if pathlib.Path(path).suffix == '.ipynb':
        with open(path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=None, kernel_type='python3')
        ep.preprocess(nb)
    else:
        runpy.run_path(str(path))
