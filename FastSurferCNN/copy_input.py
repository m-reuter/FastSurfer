# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Put the input images into the subject directory: an archival copy, and the rawavg the surfaces need.

``mri/orig/001.<ext>`` is a byte-for-byte copy of the input, in the format it arrived in, so no
header field, scale factor or data type is lost. Nothing in FastSurfer reads it back; it is there so
the subject directory records what was processed. A T2 is copied the same way, as
``mri/orig/T2raw.<ext>``.

``mri/rawavg.mgz`` is the copy the tools read. ``pctsurfcon`` builds that path itself and hardcodes
the name, so it has to be an MGH file whatever the input was. Where the input is already an .mgz it
is a symlink to the archival copy; otherwise it is converted, through ``save_image`` rather than
nibabel directly, because a plain nibabel write leaves ``fov`` at 0 and FreeSurfer reports that
field as the field of view. The T2 equivalent is ``mri/orig/T2raw.mgz``, which for an .mgz input is
the archival copy itself.

The names follow recon-all: it converts a ``-T2`` input to ``mri/orig/T2raw.mgz`` with
``--no_scale 1``, and ``samseg`` and ``-T2pial`` look for it there. 001 is historic, 001, 002 and so
on being the separate runs of one session that FreeSurfer registered and averaged into rawavg;
FastSurfer takes a single input and conforms it instead, so rawavg here is that one input rather
than an average.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from FastSurferCNN.data_loader.data_utils import save_image, storable_dtype
from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)

# both relative to the subject's mri directory, so the shapes match and neither name lies
RAWAVG_PATH = Path("rawavg.mgz")
T2_RAWAVG_PATH = Path("orig/T2raw.mgz")


def image_suffix(path: Path) -> str:
    """
    The extension to give a copy of `path`, keeping a compound one such as .nii.gz intact.

    Parameters
    ----------
    path : Path
        The file whose extension is wanted.

    Returns
    -------
    str
        The extension, including the leading dot.
    """
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] == ".gz":
        return "".join(suffixes[-2:])
    return path.suffix


def archive_input(source: Path, orig_dir: Path, stem: str = "001") -> Path:
    """
    Copy `source` into `orig_dir` as `stem` plus the source's own extension.

    The copy is byte for byte, so it is the input and not a re-encoding of it.

    Parameters
    ----------
    source : Path
        The image the user passed.
    orig_dir : Path
        The `mri/orig` directory of the subject, created if it does not exist.
    stem : str, default="001"
        The name to give the copy, without an extension.

    Returns
    -------
    Path
        The path the copy was written to.
    """
    orig_dir.mkdir(parents=True, exist_ok=True)
    destination = orig_dir / f"{stem}{image_suffix(source)}"
    if destination.resolve() == source.resolve():
        LOGGER.info(f"The input is already at {destination}, not copying it onto itself.")
        return destination
    LOGGER.info(f"Copying {source} to {destination}")
    shutil.copyfile(source, destination)
    return destination


def write_rawavg(archive: Path, rawavg: Path, source: Path | None = None) -> None:
    """
    Provide `rawavg` as an MGH file, by symlink where the input is one already and by conversion else.

    Parameters
    ----------
    archive : Path
        The archival copy of the input, which the symlink points at.
    rawavg : Path
        The `mri/rawavg.mgz` to create.
    source : Path, optional
        Where to read the voxels from, defaulting to `archive`. Passing the original input avoids
        reading the copy back, which matters when the subject directory is on slower storage than
        the input.
    """
    source = archive if source is None else source
    rawavg.parent.mkdir(parents=True, exist_ok=True)
    if archive.resolve() == rawavg.resolve():
        # an .mgz T2, whose archival copy is already at the name the tools read
        LOGGER.info(f"{rawavg} is the archival copy, nothing to convert.")
        return
    if rawavg.is_symlink() or rawavg.exists():
        rawavg.unlink()

    if image_suffix(archive) == ".mgz":
        # relative, so the subject directory stays movable; relpath rather than relative_to, which
        # refuses any layout where the archive is not below the link
        target = Path(os.path.relpath(archive, rawavg.parent))
        try:
            rawavg.symlink_to(target)
            LOGGER.info(f"Linking {rawavg} to {target}")
            return
        except OSError as error:
            LOGGER.info(f"Could not link {rawavg} ({error}), copying instead.")
            shutil.copyfile(archive, rawavg)
            return

    LOGGER.info(f"Converting {source} to {rawavg}")
    image = nib.load(source)
    data = np.asanyarray(image.dataobj)
    save_image(image.header, image.affine, data, rawavg, dtype=storable_dtype(data))


def make_parser() -> argparse.ArgumentParser:
    """Create the command line interface."""
    parser = argparse.ArgumentParser(
        description="Copy the input images into the subject directory and provide mri/rawavg.mgz.",
    )
    parser.add_argument("--t1", type=Path, required=True, help="the T1 image the user passed")
    parser.add_argument("--t2", type=Path, default=None, help="the T2 image, if one was passed")
    parser.add_argument("--sd", type=Path, required=True, help="the subjects directory")
    parser.add_argument("--sid", required=True, help="the subject id")
    return parser


def main(t1: Path, sd: Path, sid: str, t2: Path | None = None) -> int:
    """
    Copy the inputs into the subject directory and provide rawavg.

    Parameters
    ----------
    t1 : Path
        The T1 image the user passed.
    sd : Path
        The subjects directory.
    sid : str
        The subject id.
    t2 : Path, optional
        The T2 image, if one was passed.

    Returns
    -------
    int
        0 on success.
    """
    mri_dir = sd / sid / "mri"
    # (modality, source, name for the archival copy, path of the mgz the tools read)
    inputs = [("T1", t1, "001", RAWAVG_PATH)]
    if t2 is not None:
        inputs.append(("T2", t2, "T2raw", T2_RAWAVG_PATH))

    for modality, source, _stem, _rawavg in inputs:
        if not source.is_file():
            LOGGER.error(f"The {modality} file {source} does not exist.")
            return 1

    for _modality, source, stem, rawavg in inputs:
        archive = archive_input(source, mri_dir / "orig", stem=stem)
        write_rawavg(archive, mri_dir / rawavg, source=source)
    return 0


if __name__ == "__main__":
    logging.setup_logging()
    sys.exit(main(**vars(make_parser().parse_args())))
