"""frame2json converts tinyAP dataframe records into the tune JSON format."""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd


def allowed_basal_rates_522():
    # Cribbed from Loop.
    pulses_per_unit = 20.0
    return [float(u) / pulses_per_unit for u in range(0, 701)]


def read_and_clean_data(file_like):
    """read_and_clean data """
    frame = pd.read_csv(file_like)
    frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    frame.index = frame.index.tz_convert("US/Pacific")

    # Prune columns that aren't "raw".
    # frame = frame[['sgv', 'uciXS', 'uciS', 'uciM', 'uciL', 'ubi', 'deltaipid']]
    # We separate out humalog and fiasp, and also convert to U.
    frame["insulin_humalog"] = frame["deltaipid"]
    frame["insulin_fiasp"] = frame["ubi"]
    frame = frame.drop(columns=["deltaipid", "ubi"])
    frame = frame.loc["2019-09-20":]
    # Resample the frame to make sure we have samples for every period.

    # Cleanup: filter out data points that are within retractions.
    for col in ["insulin_fiasp", "uciXS", "uciS", "uciM", "uciL"]:
        frame.loc[frame[col] < 0.0] = math.nan

    return frame


def delta(array):
    shifted = np.zeros_like(array)
    shifted[1:] = array[: len(array) - 1]
    return array - shifted


def rle(array):
    list = np.array(array).tolist()
    out = []
    i = 0
    for j in range(1, len(list)):
        if list[i] == list[j]:
            continue
        n = j - i
        if n > 1:
            out.append((list[i], n))
        else:
            out.append(list[i])
        i = j
    return out


def encode(array):
    return delta(array).tolist()


def timeline(ctype, params, series):
    series = series[series != 0]
    series = series[np.isfinite(series)]
    index = encode(series.index.astype(np.int64) // 10 ** 9)
    values = encode(series.values)

    return {
        "type": ctype,
        "parameters": params,
        "index": index,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, nargs="?", help="file to convert")
    parser.add_argument("--output", type=str, help="output to file")

    args = parser.parse_args()
    input = args.file
    if input is None:
        input = sys.stdin

    frame = read_and_clean_data(input)

    def make_schedule(*args):
        index, values = zip(*args)
        return {
            "index": list(index),
            "values": list(values),
        }

    payload = json.dumps(
        {
            "version": 1,
            "timezone": "US/Pacific",
            # These are all for Medtronic 522.
            "minimum_time_interval": 30 * 60,
            "maximum_schedule_item_count": 48,
            "allowed_basal_rates": allowed_basal_rates_522(),
            # Sydney Humalog
            "basal_insulin_parameters": {
                "delay": 1.001781489035799 * 5.0,
                "peak": 13.034515673823211 * 5.0,
                "duration": 40.73599998325823 * 5.0,
            },
            "timelines": [
                timeline("glucose", {}, frame["sgv"]),
                timeline(
                    "insulin",
                    {"delay": 5, "peak": 65, "duration": 205},
                    frame["insulin_humalog"],
                ),
                timeline(
                    "insulin",
                    {"delay": 8, "peak": 44, "duration": 200},
                    frame["insulin_fiasp"],
                ),
                timeline(
                    "carb", {"delay": 10, "duration": 30}, frame["uciXS"]),
                timeline("carb", {"delay": 15, "duration": 60}, frame["uciS"]),
                timeline(
                    "carb", {"delay": 15, "duration": 120}, frame["uciM"]),
                timeline(
                    "carb", {"delay": 15, "duration": 180}, frame["uciL"]),
            ],
            "insulin_sensitivity_schedule": make_schedule(
                (0, 140),
                (3*60, 140),
                (6*60, 100),
                (8*60, 90),
                (12*60, 100),
                (15*60, 120),
                (22*60, 140),
            ),
            "basal_rate_schedule": make_schedule(
                (0, 0.2),
                (3*60, 0.1),
                (6*60, 0.5),
                (10*60, 0.3),
                (14*60, 0.25),
                (17*60, 0.20),
                (20*60, 0.15),
            ),
            #            "carb_ratio_schedule": {"index": [72, 120, 216], "values": [8, 15, 18],},
            "carb_ratio_schedule": make_schedule(
                (0, 15),
                (7*60, 8),
                (9*60, 14),
                (19*60, 20),
            ),
            "tuning_limit": 0.35,
        }
    )

    if args.output is None:
        sys.stdout.write(payload)
    else:
        with open(args.output) as file:
            file.write(payload)


if __name__ == "__main__":
    main()
