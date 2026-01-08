# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import asyncio
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import StdioTransport
import json
import os

async def log_handler(msg: LogMessage):
    print("LOG:", msg.data)

async def main():
    transport = StdioTransport(
        command="python",
        args=["smlp_mcp_server.py"]
    )

    client = Client(transport, log_handler=log_handler)
    async with client:
        tools = await client.list_tools()
        print("Tools:", tools)

        command_schema = {
            "analytics_mode": "train",
            "log_files_prefix": "TestMCP",
            "labeled_data": "/home/zurabk/smlp/repo/smlp/regr_smlp/data/smlp_toy_num_resp_mult.csv",
            "new_data" : None,
            "output_directory": "/home/zurabk/smlp/repo/smlp/regr_smlp/code",
            "split_test": 0.2,
            "features": "x,p1,p2",
            "response": "y1",
            "model": "dt_sklearn",
            "interactive_plots":False,
        }

        #result = await client.call_tool("add", {"a": 5, "b": 7})
        result = await client.call_tool("run_smlp_tool", {'params': command_schema})
        
        # Pretty print only the structured content
        print("\n=== Structured Content ===")
        print(json.dumps(result.structured_content, indent=2))

        # Or print just stdout_log
        print("\n=== SMLP stdout ===")
        print(result.structured_content.get("stdout_log", ""))
        print("====================\n")

    await client.close()
    
if __name__ == "__main__":
    asyncio.run(main())
