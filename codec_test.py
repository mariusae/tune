import unittest

import numpy as np
import pandas as pd

import codec


class TestCodec(unittest.TestCase):
    def test_decode(self):
        frame = codec.Request.fromdict(
            {
                "version": 1,
                "timezone": "US/Pacific",
                "timelines": [
                    {
                        "type": "glucose",
                        "parameters": {"meta": "yes",},
                        "index": [1576701990, 300, 300],
                        "values": [100, 10, -15],
                        "durations": [0, 0, 600],
                    }
                ],
            }
        )
        self.assertEqual(frame.timezone, "US/Pacific")
        self.assertEqual(len(frame.timeseries), 1)
        series = frame.timeseries[0]
        self.assertEqual(series.ctype, "glucose")
        self.assertEqual(series.meta, {"meta": "yes"})
        np.testing.assert_array_equal(series.series.values, [100, 110, 47.5, 47.5])
        self.assertTrue(
            (
                series.series.index
                == pd.DatetimeIndex(
                    [
                        "2019-12-18 20:46:30+00:00",
                        "2019-12-18 20:51:30+00:00",
                        "2019-12-18 20:56:30+00:00",
                        "2019-12-18 21:01:30+00:00",
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq=None,
                )
            ).all()
        )

    # print(column.series().asfreq('5min'))


if __name__ == "__main__":
    unittest.main()
