# server.py
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Add two numbers"""
    return a - b


@mcp.tool()
def append(a: str) -> int:
    """Add two numbers"""
    return "test -- " + a


if __name__ == "__main__":
    mcp.run()
