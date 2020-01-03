import json
import logging

from flask import Flask, jsonify, request

import codec
import model

app = Flask(__name__)


def response(request, model):
    return codec.Response(
        version=1,
        timezone="unavailable",
        insulin_sensitivity_schedule=codec.Schedule.fromtuple(
            model.params["insulin_sensitivity_schedule"]),
        carb_ratio_schedule=codec.Schedule.fromtuple(
            model.params["carb_ratio_schedule"]),
        basal_rate_schedule=codec.Schedule.fromtuple(
            model.params["basal_rate_schedule"]),
        training_loss=-1.,
    )


@app.route("/standard", methods=["POST"])
def standard():
    payload = request.json
    if payload is None:
        return "invalid payload", 400

    user_request = codec.Request.fromdict(payload)
    fitted_model = model.fit(user_request)

    return jsonify(response(user_request, fitted_model).todict())


@app.route("/sydney", methods=["POST"])
def sydney():
    payload = request.json
    if payload is None:
        return "invalid payload", 400

#    with open("/tmp/request.json", "w") as file:
#        json.dump(payload, file)

    user_request = codec.Request.fromdict(payload)

    # TODO: merge this once we have enough Loop data.

    with open("sydney2019-11-20.json") as file:
        canned_payload = json.load(file)
        canned_request = codec.Request.fromdict(canned_payload)
        user_request.timeseries = canned_request.timeseries

    fitted_model = model.fit(user_request)

    return jsonify(response(user_request, fitted_model).todict())


@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(
            e
        ),
        500,
    )


if __name__ == "__main__":
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
