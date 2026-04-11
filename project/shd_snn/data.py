"""Dataset download and sparse batch generation for the SHD dataset."""

import gzip
import hashlib
import os
import shutil
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import torch


def get_shd_dataset(cache_dir: str) -> str:
    """Download and decompress the SHD dataset. Returns the data directory path."""
    base_url = "https://zenkelab.org/datasets"
    cache_dir = os.path.abspath(cache_dir)

    response = urllib.request.urlopen(f"{base_url}/md5sums.txt")
    data = response.read()
    lines = data.decode("utf-8").split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}

    files = ["shd_train.h5.gz", "shd_test.h5.gz"]
    for fn in files:
        origin = f"{base_url}/{fn}"
        hdf5_file_path = _get_and_gunzip(
            origin, fn, md5hash=file_hashes[fn], cache_dir=cache_dir,
        )
        print(f"Available at: {hdf5_file_path}")

    return cache_dir


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
    """Generate sparse tensor batches from HDF5 spike data.

    Args:
        X: HDF5 group with 'times' and 'units' datasets
        y: HDF5 dataset of labels
        batch_size: Number of samples per batch
        nb_steps: Number of time bins
        nb_units: Number of input units
        max_time: Upper time limit for binning
        device: Torch device
        shuffle: Whether to shuffle sample order
    """
    labels_ = np.array(y, dtype=int)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))

    firing_times = X["times"]
    units_fired = X["units"]

    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    for counter in range(number_of_batches):
        batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]

        coo = [[], [], []]
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units]))
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)


# --- internal helpers for downloading ---


def _get_and_gunzip(origin, filename, md5hash=None, cache_dir=None):
    gz_file_path = _get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir)
    hdf5_file_path = gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(
        hdf5_file_path
    ):
        print(f"Decompressing {gz_file_path}")
        with gzip.open(gz_file_path, "r") as f_in, open(hdf5_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def _validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"
    return str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash)


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    if (algorithm == "sha256") or (algorithm == "auto" and len(algorithm) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _get_file(
    fname,
    origin,
    md5_hash=None,
    file_hash=None,
    hash_algorithm="auto",
    cache_dir=None,
):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".data-cache")
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"

    os.makedirs(cache_dir, exist_ok=True)

    fpath = os.path.join(cache_dir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None and not _validate_file(
            fpath, file_hash, algorithm=hash_algorithm
        ):
            print(
                f"A local file was found, but it seems to be incomplete or outdated "
                f"because the {hash_algorithm} file hash does not match the original value "
                f"of {file_hash} so we will re-download the data."
            )
            download = True
    else:
        download = True

    if download:
        print("Downloading data from", origin)
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(f"URL fetch failure on {origin}: {e.code} -- {e.msg}") from e
            except URLError as e:
                raise Exception(f"URL fetch failure on {origin}: {e.errno} -- {e.reason}") from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    return fpath
