from flask import Flask
from flask import request
from flask import make_response
import logging
import os
import sys
from snowflake.snowpark import Session
from agent_gateway import Agent
from agent_gateway.tools.snowflake_tools import (
    CortexSearchTool,
)

SERVICE_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVICE_PORT = os.getenv("SERVER_PORT", 8080)


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(name)s [%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    return logger


logger = get_logger("echo-service")

app = Flask(__name__)


@app.get("/healthcheck")
def readiness_probe():
    return "I'm ready!"


@app.post("/echo")
def echo():
    """
    Main handler for input data sent by Snowflake.
    """
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

    connection_parameters = connection_parameters | {
        "host": os.getenv("SNOWFLAKE_HOST"),
        "authenticator": "oauth",
    }

    with open("/snowflake/session/token") as token_file:
        connection_parameters["token"] = token_file.read()
        logger.info("Token read from file")

    connection = Session.builder.configs(connection_parameters).create()

    logger.info("Snowflake session created successfully")

    # Define your tools
    tools = [
        CortexSearchTool(
            service_name="SEC_SEARCH_SERVICE",
            service_topic="Snowflake's business,product offerings,and performance",
            data_description="Snowflake annual reports",
            retrieval_columns=["CHUNK"],
            snowflake_connection=connection,
        ),
    ]

    agent = Agent(snowflake_connection=connection, tools=tools, memory=False)
    logger.info("Agent initialized successfully")
    message = request.json
    logger.debug(f"Received request: {message}")

    if message is None or not message["data"]:
        logger.info("Received empty message")
        return {}

    input_rows = message["data"]
    logger.info(f"Received {len(input_rows)} rows")

    output_rows = [[row[0], get_echo_response(row[1])] for row in input_rows]
    logger.info(f"Produced {len(output_rows)} rows")

    results = agent.acall("What is the stock price of Apple?")

    logger.info(f"Agent call completed with results: {results}")

    response = make_response({"data": output_rows})
    response.headers["Content-type"] = "application/json"
    logger.debug(f"Sending response: {response.json}")
    return response


def get_echo_response(input):
    return input


if __name__ == "__main__":
    app.run(host=SERVICE_HOST, port=SERVICE_PORT)
