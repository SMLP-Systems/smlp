# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from fastmcp import FastMCP
from typing import Optional
import logging
import contextlib
import sys
import io

from smlp_py.smlp_flows import SmlpFlows

# Create MCP server instance
mcp = FastMCP("SmlpServer")

#@mcp.tool()
#async def add(a: int, b: int) -> int:
#    """Add two integers."""
#    return a + b


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
