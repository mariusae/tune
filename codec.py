"""codec implements the column format described in this project's README.md"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import datetime

import numpy as np
import pandas as pd


class VersionError(Exception):
    def __init__(self, version):
        super().__init__(f"unsupported version {version}")


class MissingFieldError(Exception):
    def __init__(self, field):
        super().__init__(f'missing field: "{field}"')


@dataclass
class Timeseries:
    """Timeseries represents a single timeseries. A timeseries is a
    time-indexed Pandas series with a type ("basal", "insulin", "carb")
    and metadata (e.g., parameters for carb or insulin curves.)"""

    ctype: str
    meta: dict
    series: pd.Series


@dataclass
class Schedule:
    """Schedule represents a time-indexed parameter schedule."""

    index: List[int]
    values: List[float]

    def reindexed(self, minutes):
        return np.array(self.index) // minutes

    def fromtuple(tup):
        index, values = tup
        return Schedule(index, values)

    def fromdict(d):
        return Schedule(d["index"], d["values"])

    def todict(self):
        return {
            "index": self.index,
            "values": self.values,
        }


@dataclass
class Request:
    """Request represents a tuning request. It includes the
    relevant timeseries as well as other data received from the tuning
    request."""

    # Timezone contains the user's timezone for the data in the
    # frame. This scheme does not yet support timezone mobility;
    # as such, timezones should be contained in its own timeline.
    timezone: str

    timeseries: List[Timeseries]

    minimum_time_interval: Optional[int] = None
    maximum_schedule_item_count: Optional[int] = None
    allowed_basal_rates: Optional[List[float]] = None

    basal_insulin_parameters: Dict[str, float] = field(
        default_factory=lambda: {"delay": 5, "peak": 65, "duration": 205, }
    )

    insulin_sensitivity_schedule: Optional[Schedule] = None
    carb_ratio_schedule: Optional[Schedule] = None
    basal_rate_schedule: Optional[Schedule] = None

    # If specified, limits the amount of deviation allowed from
    # the above parameters.
    tuning_limit: Optional[float] = None

    hyper_params: Dict[str, Any] = field(default_factory=lambda: {})

    # The set of parameters to tune. If it is not specified,
    # all parameters are tuned.
    tune_parameters: Optional[Set[str]] = None

    def fromdict(payload) -> "Request":
        """Decode a JSON payload into a Request."""
        if payload.get("version") is None:
            raise MissingFieldError("version")
        if payload["version"] != 1:
            raise VersionError(payload["version"])

        timezone = payload.get("timezone")
        if timezone is None:
            raise MissingFieldError("timezone")

        # iOS has the habit of sending timezone offsets.
        if timezone.startswith("GMT-") or timezone.startswith("GMT+"):
            delta = datetime.timedelta(
                hours=int(timezone[4:6]), minutes=int(timezone[6:8]))
            if timezone[3] == "-":
                delta = -delta
            timezone = datetime.timezone(delta)

        raw_timelines = payload.get("timelines")
        if raw_timelines is None:
            raise MissingFieldError("timelines")

        minimum_time_interval = payload.get("minimum_time_interval")
        maximum_schedule_item_count = payload.get(
            "maximum_schedule_item_count")
        allowed_basal_rates = payload.get("allowed_basal_rates")

        insulin_sensitivity_schedule = None
        if "insulin_sensitivity_schedule" in payload:
            insulin_sensitivity_schedule = Schedule.fromdict(
                payload["insulin_sensitivity_schedule"]
            )
        carb_ratio_schedule = None
        if "carb_ratio_schedule" in payload:
            carb_ratio_schedule = Schedule.fromdict(
                payload["carb_ratio_schedule"])
        basal_rate_schedule = None
        if "basal_rate_schedule" in payload:
            basal_rate_schedule = Schedule.fromdict(
                payload["basal_rate_schedule"])

        basal_insulin_parameters = payload.get("basal_insulin_parameters", {})

        timeseries = []
        for index, timeline in enumerate(raw_timelines):
            series_type = timeline["type"]
            if not series_type in ["bolus", "basal", "insulin", "carb", "glucose"]:
                raise Exception(
                    f"series {index}: invalid series type {series_type}")
            params = timeline.get("parameters", {})
            index = undelta(timeline["index"])
            values = undelta(timeline["values"])
            if len(index) == 0:
                continue
            if "durations" in timeline:
                durations = undelta(timeline["durations"])
                index, values = resample(index, values, durations)
            index = pd.to_datetime(index, unit="s", utc=True)
            index = index.tz_convert(timezone)
            series = pd.Series(values, index)
            timeseries.append(Timeseries(series_type, params, series))

        tune_parameters = payload.get("tune_parameters")
        if tune_parameters is not None:
            tune_parameters = set(tune_parameters)

        return Request(
            timezone=timezone,
            timeseries=timeseries,
            minimum_time_interval=minimum_time_interval,
            maximum_schedule_item_count=maximum_schedule_item_count,
            allowed_basal_rates=allowed_basal_rates,
            basal_insulin_parameters=basal_insulin_parameters,
            insulin_sensitivity_schedule=insulin_sensitivity_schedule,
            carb_ratio_schedule=carb_ratio_schedule,
            basal_rate_schedule=basal_rate_schedule,
            tuning_limit=payload.get("tuning_limit"),
            hyper_params=payload.get("hyper_params", {}),
            tune_parameters=tune_parameters,
        )


def undelta(list):
    """Decode a delta-encoded series."""
    array = np.array(list)
    array = np.cumsum(array)
    return array


def resample(index, values, durations):
    """Resample the series provided the given durations (in seconds).
    The data are always resampled to 5 minute increments. Note that
    the returned index may have duplicate entries and also be out of
    order. However, this should present no concern as these series are
    immediately resampled."""
    index_out = []
    values_out = []
    for timestamp, value, duration in zip(index, values, durations):
        # Special case for instanteneous events: we spread it across
        # the full period. (That's the limit of the model resolution
        # anyway.)
        if duration == 0:
            duration = 300
        nperiod = (duration + 300 - 1) // 300
        for period in range(nperiod):
            index_out.append(timestamp + period * 300)
            values_out.append(value / nperiod)

    return index_out, values_out


@dataclass
class Response:
    version: int
    timezone: str

    insulin_sensitivity_schedule: Schedule
    carb_ratio_schedule: Schedule
    basal_rate_schedule: Schedule

    training_loss: Optional[float]

    def todict(self):
        d = {
            "version": self.version,
            "timezone": self.timezone,
            "insulin_sensitivity_schedule": self.insulin_sensitivity_schedule.todict(),
            "carb_ratio_schedule": self.carb_ratio_schedule.todict(),
            "basal_rate_schedule": self.basal_rate_schedule.todict(),
        }
        if self.training_loss is not None:
            d["training_loss"] = self.training_loss
        return d

    def fromdict(d):
        return Response(
            version=d["version"],
            timezone=d["timezone"],
            insulin_sensitivity_schedule=Schedule.fromdict(
                d["insulin_sensitivity_schedule"]
            ),
            carb_ratio_schedule=Schedule.fromdict(d["carb_ratio_schedule"]),
            basal_rate_schedule=Schedule.fromdict(d["basal_rate_schedule"]),
            training_loss=d.get("training_loss"),
        )
