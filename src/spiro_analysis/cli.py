import imageio.v2 as imageio
import pandas as pd


from spiro_analysis.preprocess import *
from spiro_analysis.analysis import run_analysis

import click
import tomllib

DEFAULT_ANALYSIS_CONFIG = "/configs/default_analysis.toml"

@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(DEFAULT_ANALYSIS_CONFIG)
)
def analysis(config_path: Optional[Path]):

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
            f"Default expected at: {repo_root / 'configs' / 'default.toml'}\n"
            f"Pass --config PATH to use a different file."
        )
    
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    run_analysis(cfg, config_path=config_path)

    return
 