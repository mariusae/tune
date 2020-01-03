# Loop parameter estimation

Tune implements parameter estimation for
[Loop](https://github.com/LoopKit/Loop). Tune employs a simple "basal
and bolus" model that can estimate basal rates, carb ratios, and
insulin sensitivities.

## Web service

This repository includes a Google AppEngine service (deployed at
https://tune.basal.io/) that accepts a user's historical data, trains
the model, and provides estimated parameters.

## API

The web service can expose a number of models. Each model
is named by its URI. For example, the "standard" model is reached
via the URL `https://tune.basal.io/standard`.

Model handlers accept JSON-formatted POST requests with content-type
"application/json". The JSON payload encodes a number of timelines,
indexed by (Unix) timestamp. Each timeline has a type and parameters.
Timeline values are delta encoded (the value is the difference from
the previous; the first value is the difference from 0). ~~Timeline values
may also be runlength encoded by specifying a 2-element array in place
of the value: the first element is the value; the second the number of repetitions.
Run length encoding is applied after delta encoding.~~
Missing timeline values should be omitted; in particular,
the presence of NaN values is undefined.

The JSON content should have the following layout. Timelines should
be provided for insulin delivery, glucose readings, and carbohydrate
entries.

```
{
	// Version indicates the version of the data schema to be used.
	// This informal layout document describes version 1.
	"version": 1,

	// Timezone is the timezone of all of the data. This is used to
	// compute the "time of day" in the model. (Note: currently the
	// model does not make any assumptions about, e.g., night-time vs.
	// day time, but it may in the future.)
	"timezone": "US/Pacific",

	// The smallest time interval for basal rate schedules, in seconds. (Optional.)
	"minimum_time_interval": 1800,

	// The maximum number of basal rate entries. (Optional.)
	"maximum_schedule_item_count": 48,

	// Allowable basal rate values. (Optional.)
	"allowed_basal_rates": [0, 0.05, 0.1, 0.15, 0.2, ...],

	// The parameters (see below) for basal insulin delivery.
	"basal_insulin_parameters": {
		"delay": 5.008907445178995,
		"peak": 65.17257836911605,
		"duration": 203.67999991629117
	},

	// Current parameter schedules: insulin_sensitivity_schedule,
	// basal_rate_schedule, and carb_ratio_schedule are the current
	// parameter schedules in use by the subject. Schedules are indexed
	// by time (minutes) since midnight.
	//
	// These schedules may be used to initialize model parameters. Some
	// models may attempt to "fine-tune" user-supplied parameters, while
	// others may attempt to build de novo parameter schedules.

	// The current insulin sensitivity schedule. Values are in mg/dL/U.
	"insulin_sensitivity_schedule": {
		"index": [360, 720, ...],
		"values": [140, 100, ...],
	},

	// The current carbohydrate ratio schedule. Values are in g/U.
	"carb_ratio_schedule": {
		"index": [0, 360, ...],
		"values": [25, 16, ...],
	},

	// The current basal rate schedule. Values are in U/h.
	"basal_rate_schedule": {
		"index": [360, ...],
		"values": [0.25, ...],
	}

	// Optional parameter specifying the allowable parameter tuning range.
	// If specified and greater than zero, it indicates the percentage of value
	// that should be considered the tunable range.
	"tuning_limit": 0.35,

	// Timelines contains time-indexed data for all features relevant to
	// modeling insulin and carbohydrate response.
	"timelines": [
		{
			// Basal insulin columns must specify the delay, peak, and
			// duration (all in minutes), parameterizing the insulin curves.
			// Each column value is the rate delivered at the provided time;
			// the duration given was the amount of time this basal rate was
			// delivered.
			"type": "basal",
			"parameters": {
				"delay": 6,
				"peak": 65,
				"duration": 200,
			},
			// Index is delta-encoded Unix timestamps.
			"index": [
				1576623834,
				300,
				300,
				...
			],
			// Basal insulin values are given in milliunits per hour (mU/h).
			"values": [
				500,
				100,
				-200,
				...
			],
			// Durations are given in seconds.
			"durations": [
				3600,
				120,
				...
			],
		},
		{
			// Insulin columns must specify the delay, peak, and
			// duration (all in minutes), parameterizing the insulin
			// curves. Each column value is the amount delivered at
			// the indexed time.
			"type": "insulin",
			"parameters": {
				"delay": 6,
				"peak": 65,
				"duration": 200,
			},
			// Index is delta-encoded Unix timestamps.
			"index": [
				1576623834,
				300,
				300,
				300,
				...
			],
			// Basal insulin values are given in milliunits per hour (mU/h).
			"values": [
				50,
				100,
				60,
				...
			],
		},
		{
			// Carb columns must specify the delay and duration of
			// absorption (both in minutes).
			"type": "carb",
			"parameters": {
				"delay": 15,
				"duration": 120,
			}
			"index": [
				1576623834,
				164,
				...
			],
			// Carb values are given in grams (g).
			"values": [
				10,
				-5,
				...
			]
		},
		{
			"type": "glucose"
			"index": [
				1576623834,
				300,
				300,
				...
			]
			// Glucose values are given in mg/dL.
			"values": [
				110,
				2,
				-4,
				...
			]
		}
	]
}
```

Alternative encodings may be offered in the future allowing for a
more compact representation of values. (TODO: consider allowing for
delta-encoded values in this encoding too.) The current encoding
compresses (using gzip) to about 40kB for 3 months of data for my
daughter, with deliveries (almost) every 5 minutes, and the frequency
of meals that can be attained only by 6 year-olds and ultra-marathon
runners. (Without delta encoding, the same content compresses to
about 100kB).

Once the model has been trained, the model handler returns the estimated
parameters in a JSON-formatted body (content-type "application/json") with
the following layout:

```
{
	// The version corresponding to the output. This should always
	// be the same as in the request.
	"version": 1,

	// The timezone for which the time of day indices below are relative.
	// This should always be the same as in the request.
	"timezone": "US/Pacific",

	// Modeled parameters: insulin_sensitivity_schedule,
	// carb_ratio_schedule, and basal_rate_schedule. These are as in the
	// request. Each represents a schedule of parameters, indexed by
	// time (in minutes) relative to midnight.
	//
	// Each schedule specifies an index (whose values are minutes since
	// midnight) and a value vector, whose values depends on the
	// specific value type. The entries are sorted by start time. The
	// schedule is always complete: the value for one entry is valid
	// until the next entry. The last entry may span midnight: it always
	// ends at the time of the first entry.

	"insulin_sensitivity_schedule": {
		"index": [360, 720, ...]
		// The insulin sensitivity at 360 minutes past 00:00 (i.e.,
		// at 6am) is 119, at 720 minutes past 00:00 (noon)
		// it is 130,
		"values": [119, 130, ...],
	},

	// The computed carb ratios, in g/U.
	"carb_ratio_schedule": {
		"index": [0, ...],
		"values": [17.615762468879538, ...],
	},

	// The computed basal rates, in U/h.
	"basal_rate_schedule": {
		"index": [180, ...],
		"values": [0.3, ...],
	}

	// The training loss (goodness of fit) of the above parameters. This
	// is not interpretable by the user except by relative comparison:
	// Lower values indicate a better fit.
	"training_loss": 0.7105391088965228
}
```

These formats are implemented by the python module
[codec.py](https://github.com/mariusae/tune/blob/master/codec.py).

## Development

The AppEngine frontend is a [Flask](https://www.palletsprojects.com/p/flask/)
application. The Python module `main.py` implements the tuning service API;
it can be run locally:

```
$ python3 main.py
```

The project may also be deployed as an AppEngine server. Once you
have set up an AppEngine project, you can deploy it in the usual
manner, from the present directory:

```
$ gcloud app deploy
```
