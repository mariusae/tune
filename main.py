import json
import logging

from flask import Flask, jsonify, request

import codec
import model

app = Flask(__name__)


@app.route("/standard", methods=["POST"])
def standard():
    payload = request.json
    if payload is None:
        return "invalid payload", 400

    frame = codec.Request.fromdict(payload)
    params = model.fit(frame).params

    parameters = {}
    for name in params:
        index, values = zip(*params[name])
        parameters[name] = {
            "index": list(index),
            "values": list(values),
        }

    return jsonify(
        {{"version": 1, "timezone": frame.timezone, "parameters": parameters}}
    )


@app.route("/sydney", methods=["POST"])
def sydney():
    payload = request.json
    if payload is None:
        return "invalid payload", 400

    # TODO: merge this once we have enough Loop data.

    with open("sydney2019-11-20.json") as file:
        payload = json.load(file)
    frame = codec.Request.fromdict(payload)
    params = model.fit(frame).params

    parameters = {}
    for name in params:
        index, values = zip(*params[name])
        parameters[name] = {
            "index": list(index),
            "values": list(values),
        }

    return jsonify({"version": 1, "timezone": frame.timezone, "parameters": parameters})


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
