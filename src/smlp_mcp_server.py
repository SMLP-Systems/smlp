# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

# Defines a FastMCP server with one tool: run_smlp_tool.
# Exposes it via stdio when run directly (mcp.run(transport="stdio")).
#
# To test it standlone:
# 1) Run the server directly (should start and wait for stdio)
# PYTHONPATH=src python -m smlp_mcp_server
# 2) Try a bare import from a Python REPL:
# PYTHONPATH=src python -c "import smlp_py.smlp_flows as m; print('OK', m.__file__)"


import logging
import contextlib
import sys
import io

from fastmcp import FastMCP
#from fastmcp import Server, tool


# --- path bootstrap: make sure this script finds modules in src/ ---
from pathlib import Path
import sys as _sys

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent       # .../repo/smlp/src
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from smlp_py.smlp_flows import SmlpFlows


#@mcp.tool()
#async def add(a: int, b: int) -> int:
#    """Add two integers."""
#    return a + b



# Create MCP server instance
mcp = FastMCP("SmlpServer")

@mcp.tool()
async def run_smlp_tool(params: dict) -> dict:
    """
    Run SMLP with CLI-style parameters. 
    Params dict should match CLI args (e.g., {'labeled_data': '...', 'model': 'dt_caret'}).
    """
    print('===================run_smlp_tool params', type(run_smlp_tool), run_smlp_tool, file=sys.stderr, flush=True)
    # Convert dict to CLI-like argv list
    argv = ["smlp"]
    for k, v in params.items():
        flag = f"--{k}"
        argv.extend([flag, str(v)])
        #if isinstance(v, bool):
        #    if v:
        #        argv.append(flag)
        #else:
        #    argv.extend([flag, str(v)])
    print('argv in run_smlp_tool', argv)
    
    try:
        smlp = SmlpFlows(argv)
        smlp.smlp_flow()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # No asyncio.run() here — FastMCP manages the loop
    mcp.run(transport="stdio")

    
'''

from fastmcp import FastMCP
from typing import Optional


# Create MCP server instance
mcp = FastMCP("SmlpServer")

@mcp.tool()
async def run_smlp_tool(
    analytics_mode: str,
    log_files_prefix: str,
    labeled_data: Optional[str] = None,
    new_data: Optional[str] = None,
    output_directory: str = None,
    split_test: Optional[float] = 0.2,
    features: Optional[str] = None,
    response: Optional[str] = None,
    model: Optional[str] = None,
    interactive_plots: bool = False
):
    """
    Runs SMLP with given parameters in MCP-safe mode.
    - Redirects all stdout to buffer to avoid MCP JSON corruption.
    - Forces logging to stderr (client log handler will show these).
    - Returns captured stdout in response for debugging.
    """

    # --- Configure logging for both this function and SMLP ---
    logging.basicConfig(
        stream=sys.stderr,
        #level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Attach SMLP's logger to same config (if SMLP uses logging.getLogger)
    smlp_logger = logging.getLogger("smlp")
    smlp_logger.setLevel(logging.DEBUG)

    logging.debug("Starting run_smlp_tool ...")

    # Build argv for SMLP
    argv = ["dummy_argv[0]", "--analytics_mode", analytics_mode, "--log_files_prefix", log_files_prefix]
    if labeled_data:
        argv += ["--labeled_data", labeled_data]
    if new_data:
        argv += ["--new_data", new_data]
    if split_test is not None:
        argv += ["--split_test", str(split_test)]
    if features:
        argv += ["--features", features]
    if response:
        argv += ["--response", response]
    if model:
        argv += ["--model", model]
    if output_directory:
        argv += ["--output_directory", output_directory]
    if interactive_plots:
        argv.append("--interactive_plots")

    logging.debug(f"Final argv: {argv}")
    print("argv", argv, flush=True)

    smlp_instance = SmlpFlows(argv)

    # --- Capture stdout from SMLP run ---
    stdout_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer):
        try:
            logging.debug("Calling smlp_instance.smlp_flow() ...")
            print("Calling smlp_instance.smlp_flow()", flush=True)

            smlp_instance.smlp_flow()

            logging.debug("SMLP flow completed successfully")
            print("SMLP flow completed successfully", flush=True)

            return {
                "status": "success",
                "argv": argv,
                "stdout_log": stdout_buffer.getvalue()
            }

        except Exception as e:
            logging.exception("SMLP execution failed")  # Logs traceback to stderr
            return {
                "status": "error",
                "error": str(e),
                "argv": argv,
                "stdout_log": stdout_buffer.getvalue()
            }

if __name__ == "__main__":
    # No asyncio.run() here — FastMCP manages the loop
    mcp.run(transport="stdio")
'''