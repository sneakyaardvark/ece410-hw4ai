"""Profiling script for SHD spiking neural network inference using torch.profiler."""

import argparse
import os
import resource
import statistics
import time

import h5py
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from shd_snn.data import get_shd_dataset, sparse_data_generator
from shd_snn.model import SHDModel

NB_INPUTS = 700
NB_OUTPUTS = 20
MAX_TIME = 1.4


def main():
    parser = argparse.ArgumentParser(description="Profile SHD SNN inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model.pt")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--nb-batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--nb-hidden", type=int, default=200, help="Number of hidden neurons")
    parser.add_argument("--nb-steps", type=int, default=100, help="Number of simulation time steps")
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, "data"),
        help="Directory for cached dataset files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./profiler_traces",
        help="Directory for TensorBoard profiler trace output",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SHDModel(
        nb_inputs=NB_INPUTS,
        nb_hidden=args.nb_hidden,
        nb_outputs=NB_OUTPUTS,
        nb_steps=args.nb_steps,
        device=device,
    )
    model.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    data_path = get_shd_dataset(args.data_dir)
    test_file = h5py.File(os.path.join(data_path, "shd_test.h5"), "r")
    x_test, y_test = test_file["spikes"], test_file["labels"]

    # Pre-generate batches so data loading doesn't pollute the profile
    batches = []
    for i, (x_local, _y_local) in enumerate(
        sparse_data_generator(
            x_test, y_test, args.batch_size, model.nb_steps, NB_INPUTS, MAX_TIME, device,
            shuffle=False,
        )
    ):
        if i >= args.nb_batches:
            break
        batches.append(x_local.to_dense())

    print(f"Profiling {len(batches)} batches...")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    active = max(1, len(batches) - 2)
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for x_dense in batches:
            with torch.no_grad():
                model(x_dense, args.batch_size)
            prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print(f"\nTrace saved to {args.output_dir}/")
    print("View with: tensorboard --logdir", args.output_dir)

    # Single-sample wall-clock timing (≥10 runs)
    sample = batches[0][:1]  # one sample from first batch
    nb_runs = 10
    times = []
    for _ in range(nb_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(sample, 1)
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    print(f"\nSingle-sample inference ({nb_runs} runs):")
    print(f"  Median: {median_s*1000:.2f} ms")
    print(f"  Throughput: {1/median_s:.1f} samples/sec")
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  Peak RSS: {peak_rss_kb / 1024:.1f} MB")

    test_file.close()


if __name__ == "__main__":
    main()
