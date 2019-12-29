import math
import unittest

import numpy as np
import pandas as pd

import codec
import model


class TestResample(unittest.TestCase):
    def test_resample(self):
        index = pd.DatetimeIndex(
            [
                "2019-12-18 20:46:30+00:00",
                "2019-12-18 20:46:31+00:00",
                "2019-12-18 20:50:00+00:00",
                "2019-12-18 20:54:00+00:00",
                "2019-12-18 20:55:00+00:00",
            ],
            dtype="datetime64[ns, UTC]",
        )

        timeseries = [
            codec.Timeseries(
                "insulin",
                {"delay": 10, "peak": 60, "duration": 60 * 3},
                pd.Series([1, 11, 200, 20, 300], index),
            ),
            codec.Timeseries(
                "glucose",
                {},
                pd.Series(
                    [120, 110],
                    pd.DatetimeIndex(
                        ["2019-12-18 20:46:00+00:00", "2019-12-18 20:55:00+00:00",]
                    ),
                ),
            ),
        ]

        frame = codec.Request("US/Pacific", timeseries)
        frame = model.resample(frame)

        self.assertEqual(frame.timezone, "US/Pacific")
        self.assertEqual(len(frame.timeseries), 2)

        # There doesn't appear to be a straightforward way to resample
        # indices directly in Pandas.
        resampled_index = pd.Series(0, index).resample("5min").bfill().index
        for col in frame.timeseries:
            self.assertTrue((col.series.index == resampled_index).all())

            if col.ctype == "insulin":
                np.testing.assert_array_equal(col.series.values, [12, 220, 300])
            elif col.ctype == "glucose":
                np.testing.assert_array_equal(col.series.values, [120, math.nan, 110])
            else:
                self.fail(f"invalid column type {col.ctype}")


class CurveTest:
    def ctype(self):
        pass

    def meta(self):
        pass

    def apply_curve(self, column):
        pass

    def assertCurve(self, curve):
        self.assertAlmostEqual(np.sum(curve), 1.0, 2)

        increasing = True
        for (i,) in np.ndindex(curve.shape):
            if i < 2:
                continue
            if increasing and curve[i] < curve[i - 1]:
                increasing = False
            if not increasing:
                self.assertTrue(curve[i] <= curve[i - 1])

    def curve_for_deliveries(self, deliveries):
        index = pd.date_range("12/1/2019", periods=model.Whoriz * 2, freq=model.Period)
        values = np.zeros_like(index, dtype=np.float64)
        for (when, value) in deliveries:
            values[when] = value
        return pd.Series(values, index)

    def timeseries_for_deliveries(self, ctype, meta, deliveries):
        return codec.Timeseries(ctype, meta, self.curve_for_deliveries(deliveries))

    def test_apply_curve(self):
        column = self.timeseries_for_deliveries(
            self.ctype(), self.meta(), [(model.Whoriz, 1)]
        )
        curve = self.apply_curve(column)

        self.assertEqual(column.series.shape, curve.shape)
        self.assertTrue((column.series.index == curve.index).all())

        # The first Whoriz-1 values should be NaNs because we don't
        # have enough data to compute insulin activity for them.
        np.testing.assert_array_equal(
            np.zeros(model.Whoriz - 1) * math.nan, curve.values[: model.Whoriz - 1],
        )

        curve = curve.values[model.Whoriz - 1 :]
        # Invariants to test:
        # - the first delay values should be zero
        # - the curve of the array should sum to ~1 (all insulin accounted for)
        # - the curve should be monotonically increasing,
        #   and then monotonically decreasing
        delay = self.meta()["delay"] // 5
        np.testing.assert_array_equal(np.zeros(delay), curve[:delay])
        self.assertCurve(curve)

    def test_apply_curves_compose(self):
        """Test to ensure that insulin curves compose."""
        index = pd.date_range("12/1/2019", periods=model.Whoriz * 2, freq=model.Period)

        def curve(deliveries):
            return self.apply_curve(
                self.timeseries_for_deliveries(self.ctype(), self.meta(), deliveries)
            )

        deliveries = [(model.Whoriz, 1.0), (model.Whoriz + 12, 2.0)]
        np.testing.assert_array_equal(
            curve(deliveries), curve(deliveries[:1]) + curve(deliveries[1:])
        )


class TestInsulinCurve(unittest.TestCase, CurveTest):
    def ctype(self):
        return "insulin"

    def meta(self):
        return {"delay": 10, "peak": 30, "duration": 120}

    def apply_curve(self, column):
        return model.apply_insulin_curve(column)


class TestCarbCurves(unittest.TestCase, CurveTest):
    def ctype(self):
        return "carb"

    def meta(self):
        return {"delay": 15, "duration": 60}

    def apply_curve(self, column):
        return model.apply_carb_curve(column)


def make_test_frame():
    index = pd.date_range("12/1/2019", periods=12 * 24, freq=model.Period)
    zeros = np.zeros_like(index, dtype=np.float64)

    timeseries = [
        codec.Timeseries(
            "glucose", {}, pd.Series(125, pd.DatetimeIndex(["2019-12-01 06:36:04"]))
        ),
        codec.Timeseries("glucose", {}, pd.Series(zeros + 100, index)),
        codec.Timeseries(
            "insulin",
            {"delay": 5, "peak": 55, "duration": 180},
            pd.Series(zeros + 10, index),
        ),
        codec.Timeseries(
            "insulin",
            {"delay": 5, "peak": 60, "duration": 180},
            pd.Series(1000, pd.DatetimeIndex(["2019-12-01 06:05:00"])),
        ),
        codec.Timeseries(
            "carb",
            {"delay": 15, "duration": 60},
            pd.Series(
                [0, 40, 10, 4, 80],
                pd.DatetimeIndex(
                    [
                        "2019-12-01 00:00:00",
                        "2019-12-01 08:08:12",
                        "2019-12-01 08:08:30",
                        "2019-12-01 12:14:00",
                        "2019-12-01 16:11:00",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
        ),
    ]
    return codec.Request(timezone="US/Pacific", timeseries=timeseries)

class FrameTest(unittest.TestCase):
    def test_make_frame(self):
        frame = make_test_frame()
        frame = model.resample(frame)
        frame = model.make_pandas_frame(frame)

        self.assertAlmostEqual(frame["insulin"].sum(skipna=True), 3.166, 2)
        self.assertEqual(frame["glucose"].sum(), 28825)
        self.assertAlmostEqual(frame["carb"].sum(), 134, 1)


class ModelTest(unittest.TestCase):
    def test_model(self):
        frame = make_test_frame()
        m = model.fit(frame)


class IndexToIntervalsTest(unittest.TestCase):
    def test_index_to_intervals(self):
        cases = [
            ([0], [[(0, 288)]]),
            ([50], [[(0, 288)]]),
            ([0, 200], [[(0, 200)], [(200, 288)]]),
            ([50, 200], [[(50, 200)], [(200, 288), (0, 50)]]),
        ]
        for (index, intervals) in cases:
            self.assertEqual(model.index_to_intervals(index), intervals)


class ParameterPackingTest(unittest.TestCase):
    def test_pack_params(self):
        indexed_params = [
            ([0, 40, 280], ["a", "b", "c"]),
            ([0], ["justone"]),
        ]

        packed_params, indexers, unpack = model.pack_params(indexed_params)
        self.assertEqual(np.shape(indexers), (2, 288))

        np.testing.assert_array_equal(packed_params, ["a", "b", "c", "justone"])

        for i in range(288):
            first, second = unpack(packed_params)
            value = first[indexers[0]][i]
            if i < 40:
                expected = "a"
            elif i < 280:
                expected = "b"
            else:
                expected = "c"
            self.assertEqual(value, expected)
            self.assertEqual(second[indexers[1][i]], "justone")


if __name__ == "__main__":
    unittest.main()
