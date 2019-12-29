import datetime
import json
import sys

import numpy as np

import codec
import model


def index_to_time(index):
    return datetime.time(index // 12, 5 * int(index % 12))


def main():
    body = json.load(sys.stdin)
    resp = codec.Response.fromdict(body)
    print("training loss:", resp.training_loss)

    def print_schedule(name, schedule):
        print(name + ":")
        intervals = model.index_to_intervals(np.array(schedule.index) // 5)
        schedule_time = []
        for ix, intervals in enumerate(intervals):
            for beg, end in intervals:
                beg, end = index_to_time(beg), index_to_time(end - 1)
                schedule_time.append((beg, end, schedule.values[ix]))
        schedule_time = sorted(schedule_time, key=lambda x: x[0])
        for beg, end, value in schedule_time:
            print(f"\t{beg.isoformat()}-{end.isoformat()}\t{value}")

    print_schedule("basal rates", resp.basal_rate_schedule)
    print_schedule("insulin sensitivities", resp.insulin_sensitivity_schedule)
    print_schedule("carb ratios", resp.carb_ratio_schedule)


if __name__ == "__main__":
    main()
