# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from agent_gateway.tools.snowflake_tools import (
    CortexAnalystTool,
    CortexSearchTool,
    PythonTool,
    SQLTool,
)

__all__ = ["CortexAnalystTool", "CortexSearchTool", "PythonTool", "SQLTool", "MCPTool"]


def is_fastmcp_available():
    import importlib.util

    return importlib.util.find_spec("fastmcp") is not None


def __getattr__(name):
    if name == "MCPTool":
        if is_fastmcp_available():
            from agent_gateway.tools.snowflake_tools import MCPTool

            return MCPTool
        else:
            raise ModuleNotFoundError(
                "MCPTool support requires fastmcp. Install with pip install orchestration-framework[fastmcp] or pip install fastmcp."
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
