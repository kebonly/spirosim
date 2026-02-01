import imageio.v2 as imageio
import pandas as pd


from spiro_analysis.preprocess import *
from spiro_analysis.analysis import run_analysis, view_napari_tracks
# import spiro_analysis.analysis as analysis

import click
import tomllib

DEFAULT_ANALYSIS_CONFIG = "/configs/default_analysis.toml"

def handle_config_path(config_path: Optional[Path]):
    cur = Path.cwd().resolve()
    repo_root = Path.cwd()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            repo_root = p
    
    if config_path is None:
        config_path = repo_root / "configs" / "default_analysis.toml"
    else:
        config_path = config_path.expanduser()

    if not config_path.exists():
        raise click.ClickException(
            f"Config not found: {config_path}\n"
            f"Default expected at: {repo_root / 'configs'}\n"
            f"Pass --config PATH to use a different file."
        )
    
    return config_path

@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_ANALYSIS_CONFIG)
)
def analysis(config_path: Optional[Path]):
    config_path = handle_config_path(config_path=config_path)

    
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    run_analysis(cfg, config_path=config_path)

    return

@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_ANALYSIS_CONFIG)
)
def view_tracks(config_path: Optional[Path]):
    print('hi')
    config_path = handle_config_path(config_path=config_path)

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    view_napari_tracks(cfg)

    