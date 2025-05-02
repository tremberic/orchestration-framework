import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from snowflake.snowpark import Session
from agent_gateway import Agent
from agent_gateway.tools.snowflake_tools import (
    CortexSearchTool,
)
from agent_gateway.tools.utils import _determine_runtime
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up agent and initializing resources")
    try:
        # Initialize Snowflake connection
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        }

        if _determine_runtime():
            logger.info("In Snowflake.")
            connection_parameters = connection_parameters | {
                "host": os.getenv("SNOWFLAKE_HOST"),
                "authenticator": "oauth",
            }
            with open("/snowflake/session/token") as token_file:
                connection_parameters["token"] = token_file.read()
                logger.info("Token read from file")
        else:
            logger.info("Not in Snowflake.")
            connection_parameters = connection_parameters | {
                "user": os.getenv("SNOWFLAKE_USER"),
                "password": os.getenv("SNOWFLAKE_PASSWORD"),
            }
        logger.info("Connecting")
        connection = Session.builder.configs(connection_parameters).create()
        logger.info("Connected")

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

        logger.info("Tools defined")

        # Initialize the agent once
        app.state.agent = Agent(
            snowflake_connection=connection, tools=tools, memory=False
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.info(f"Failed to initialize agent: {str(e)}")

    yield  # Application runs here

    # Shutdown: Clean up resources when the application is shutting down
    logger.info("Shutting down agent and cleaning up resources")


# Pass the lifespan function to FastAPI
app = FastAPI(title="Agent Gateway API", lifespan=lifespan)


class QueryResponse(BaseModel):
    output: str
    sources: Optional[Any] = None


class QueryRequest(BaseModel):
    request: str


@app.post("/query")
async def process_query(query: QueryRequest, request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    logger.debug(f"Received request: {query.request}")
    results = await agent.acall(query.request)

    return results


@app.get("/health")
async def health_check():
    agent = getattr(request.app.state, "agent", None)
    return {
        "status": "healthy" if agent is not None else "agent not initialized",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
