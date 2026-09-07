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
``mri/orig/T2.001.<ext>``.

``mri/rawavg.mgz`` is the copy the tools read. ``pctsurfcon`` builds that path itself and hardcodes
the name, so it has to be an MGH file whatever the input was. Where the input is already an .mgz it
is a symlink to the archival copy; otherwise it is converted, through ``save_image`` rather than
nibabel directly, because a plain nibabel write leaves ``fov`` at 0 and FreeSurfer reports that
field as the field of view. A T2 gets the same pair, ``mri/T2.rawavg.mgz``.

The name 001 is historic: 001, 002 and so on were the separate runs of one session, which FreeSurfer
registered and averaged into rawavg. FastSurfer takes a single input and conforms it instead, so
rawavg here is that one input rather than an average.
"""

import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from FastSurferCNN.data_loader.data_utils import save_image, storable_dtype
from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)

RAWAVG_NAME = "rawavg.mgz"
T2_RAWAVG_NAME = "T2.rawavg.mgz"


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


def write_rawavg(source: Path, rawavg: Path) -> None:
    """
    Provide `rawavg` as an MGH file, by symlink where `source` is one already and by conversion else.

    Parameters
    ----------
    source : Path
        The archival copy of the input.
    rawavg : Path
        The `mri/rawavg.mgz` to create.
    """
    rawavg.parent.mkdir(parents=True, exist_ok=True)
    if rawavg.is_symlink() or rawavg.exists():
        rawavg.unlink()

    if image_suffix(source) == ".mgz":
        # relative, so the subject directory stays movable
        target = source.relative_to(rawavg.parent)
        try:
            rawavg.symlink_to(target)
            LOGGER.info(f"Linking {rawavg} to {target}")
            return
        except OSError as error:
            LOGGER.info(f"Could not link {rawavg} ({error}), copying instead.")
            shutil.copyfile(source, rawavg)
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
    # (modality, source, name for the archival copy, name for the mgz the tools read)
    inputs = [("T1", t1, "001", RAWAVG_NAME)]
    if t2 is not None:
        inputs.append(("T2", t2, "T2.001", T2_RAWAVG_NAME))

    for modality, source, _stem, _rawavg in inputs:
        if not source.is_file():
            LOGGER.error(f"The {modality} file {source} does not exist.")
            return 1

    for _modality, source, stem, rawavg in inputs:
        copy = archive_input(source, mri_dir / "orig", stem=stem)
        write_rawavg(copy, mri_dir / rawavg)
    return 0


if __name__ == "__main__":
    logging.setup_logging()
    sys.exit(main(**vars(make_parser().parse_args())))
