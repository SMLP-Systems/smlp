# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from fastapi import FastAPI, Request
from pydantic import BaseModel
import datetime, json, traceback

from smlp_agent import SmlpAgent, LLMInterpreter, SmlpSpecGenerator, SmlpExecutor

app = FastAPI()


# This module exposes SMLP Agent as a microservice. 
# External tools or agents can call it via HTTP.

'''
This command runs your FastAPI server:
uvicorn api_smlp_agent:app --reload

Breakdown:
--uvicorn: lightweight ASGI web server.
--api_smlp_agent: the Python file (no .py).
--app: the FastAPI instance inside that file.
--reload: enables auto-reload on code changes.

Invokation via curl:
1st shell: python -m uvicorn api_smlp_agent:app --reload
Whwn above is running, there are a few options:
1. curl command: in a 2nd shell: curl -X POST http://127.0.0.1:8000/agent/text -H "Content-Type: application/json" -d '{"query": "run rag on toy pdf and answer if a paper on SMLP is published"}'   ### takes the query from command line directly
2. curl command: in a 2nd shell: curl -X POST http://127.0.0.1:8000/agent/text  -H "Content-Type: application/json" --data  @../regr_smlp/queries/few_shot_query.json   ### loads query from json file 
3. Swagger GUI: instead of the curl command, invoke query from Swagger WEB interface -- pste query in the slot for that, then execute, logs are seen in the 1st shell. Example query for SMLP agent: { "query": "Run rag on SMLP paper using deepseek" }
4. Via a dedicated chatbot -- SMLP ChatBot (not implemented)
'''

'''
TODO 
--Replace the dummy LLMInterpreter.plan_from_text() with actual OpenAI/Ollama inference.
--Add memory or vector DB context.
--Add more task types.
--Add Chat history or memory via LangChain, Redis, or vector DB
--Add authentication
--Add streaming logs to UI
--Add access control or admin checks
--Create SMLP chat bot
--Ollama fallback if OpenAI fails (!!!!! likely not needed, not using OpenAI for now, it is not free)
--LangChain or HuggingFace integration
--Support for Claude or Groq
'''


print("!!!!! WELCOME TO SMLP AGENT API !!!!!", flush=True)
# Initialize the agent
# TODO !!! hard coded: which model to use -- Ollama vs OpenAI
use_ollama = True
if use_ollama:
    provider="ollama"
    model_name="mistral"
else:
    provider="openai"
    model_name="gpt-3.5-turbo"

print(f"""Initializing SMLP Agent with provider {provider}, model {model_name}""", flush=True)
agent = SmlpAgent(
    llm_interpreter=LLMInterpreter(provider=provider, model_name=model_name),
    spec_generator=SmlpSpecGenerator(),
    executor=SmlpExecutor()
)
print(f"""SMLP Agent was initialized with provider {provider}, model {model_name}""", flush=True)
      
# Input schema for natural language queries
class TextQuery(BaseModel):
    query: str



####################### GET methods ################################

# GET: Agent health check. Useful for Developers, dashboards.
# A "read-only" endpoints, safe to expose if you're running in a secure/internal environmen
# TODO !!! For now, this simply confirms FastAPI is up; and always returns "status": "running".
# In production, one can:
# --Add actual checks (e.g., is SMLP responsive, last error time, etc.)
# --Ping dependencies like OpenAI, Ollama, etc.
@app.get("/agent/status")
async def get_status():
    """
    Returns status and uptime (for health checks or monitoring tools).
    """
    return {"status": "running", "uptime": datetime.datetime.now().isoformat()}

# GET: Show last N events from agent logs. Useful for DevOps, developer UI.
# A "read-only" endpoints, safe to expose if you're running in a secure/internal environmen
@app.get("/agent/logs")
async def get_logs(n: int = 10):
    """
    Returns the last n entries from the log file (default 10).
    """
    with open("smlp_agent_log.jsonl") as f:
        lines = f.readlines()[-n:]
    return {"last_events": [json.loads(l) for l in lines]}


####################### POST methods ################################

# POST: Text Command (user gives plain English task). This is a core functionality of SMLP agent.
@app.post("/agent/text")
async def handle_text_command(input: TextQuery):
    """
    Accepts a user query as text, uses LLM to create a task plan and spec, then executes via SMLP.
    """
    try:
        response = agent.run_text_command(input.query)
        print('>>> SMLP Agent execution completed', flush=True)
        return {"output": response}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# POST: Structured JSON task. This toot is a core functionality of SMLP agent.
@app.post("/agent/task")
async def handle_structured_task(req: Request):
    """
    Accepts a pre-structured SMLP task spec as JSON and runs it directly.
    """
    try:
        data = await req.json()
        response = agent.run_task_command(data)
        return {"output": response}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# POST: Dry-run — get plan & spec from input text. Useful for Debuggers, chatbot UIs.
# A "read-only" endpoints, safe to expose if you're running in a secure/internal environmen
@app.post("/agent/spec-preview")
async def preview_spec(query: TextQuery):
    """
    Accepts a natural language request and returns the task plan and spec (without executing).
    """
    plan = agent.llm.plan_from_text(query.query)
    spec = agent.spec_generator.generate_from_plan(plan)
    return {"plan": plan, "spec": spec}

'''
1.
This starts your API server at:http://localhost:8000
python -m uvicorn api_smlp_agent:app --reload --port 8000
Make sure it prints "Application startup complete".
2.
python -m streamlit run smlp_chatbot.py
This will open a browser window (or show a link like http://localhost:8501) to the Streamlit app.
3.
a. Paste your natural language prompt
e.g.,
Train DT model on dataset foo.csv, output y1, features x1,x2. Save to model123.
b. (Optional) Upload a few-shot prompt file
In the left sidebar, upload a .txt file with few-shot examples. Example format:

Input: Train model on foo.csv with target y1
Output: {"data": "foo.csv", "targets": ["y1"]}

Input: Use model123 for optimization with lazy strategy
Output: {"model": "model123", "opt_strategy": "lazy"}

4.
Submit and View Results
Click the "Run SMLP" button. You’ll see:
--The natural prompt you entered
--The CLI dictionary (cli_dict) parsed from it
--The actual command run
--Any output printed by SMLP (stdout, stderr)
'''
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")

    try:
        # Run SMLP agent logic
        agent_result = agent.run_chat_command(user_input)

        # Optionally structure the output
        return {
            "status": "success",
            "input": user_input,
            "cli_dict": agent_result.get("cli_dict"),
            "command": agent_result.get("command"),
            "stdout": agent_result.get("stdout"),
            "stderr": agent_result.get("stderr")
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/load_prompt")
async def load_prompt(request: Request):
    data = await request.json()
    prompt_text = data.get("prompt", "")
    
    try:
        agent.load_prompt_from_text(prompt_text)
        return {"status": "ok", "message": "Few-shot prompt updated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



'''
TODO !!!! not functional, WIP

from fastapi import Depends, HTTPException, Header

API_KEY = "my-secret-token"  # store securely in prod

# Let’s secure certain routes (e.g., PUT, DELETE) with a basic API key authentication scheme.
def authenticate(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# seure endpoints like:
@app.put("/agent/config")
async def update_config(..., auth=Depends(authenticate)):
    return {"message": "Config updated."}

@app.delete("/agent/logs")
async def clear_logs(auth=Depends(authenticate)):
    open("smlp_agent_log.jsonl", "w").close()
    return {"message": "Logs cleared."}

Clients must include: H "x-api-key: my-secret-token"
'''

'''
# TODO !!!!  @app.put(...), @app.delete(...)

Reasons we don't use @app.put(...), @app.delete(...) (yet):
--these are for data modification; not good for applcation securty 
--SMLP doesn’t currently modify stored data through this agent
--PUT and DELETE are typically used for CRUD APIs (which stands for: 
  Create → POST; Read → GET; Update → PUT / PATCH; Delete → DELETE)
--Adding them without access control risks misuse or attacks

Legitimate use cases:
--PUT /agent/config — update default model, base URL, logging level, etc.
--DELETE /agent/logs — clear logs (only for admins)
One can add these, but always with access control or admin checks.
'''

'''
TODO !!!
def fetch_data():          # Sync (blocking)
    result = http_get()    # Waits here fully

async def fetch_data():    # Async (non-blocking)
    result = await http_get_async()

'''

'''

To Run SMLP Agent with Ollama (Step-by-step)
1. Start Ollama Server
This launches Ollama's local inference server.

ollama serve
This should stay running in its own terminal.

It starts a local REST API at http://localhost:11434.

2. Pull the Model (once only)
This downloads the model (e.g. mistral) to your machine.

ollama pull mistral
You only need to do this once per model (unless you delete/update it).

3. Start SMLP Agent FastAPI Server
In another terminal (recommended), run:

python -m uvicorn api_smlp_agent:app --reload
This starts your SMLP Agent HTTP server at http://127.0.0.1:8000.

4. Make a Request to the Agent (e.g. via curl)
Now you can interact with your agent using:

curl -X POST http://127.0.0.1:8000/agent/text \
  -H "Content-Type: application/json" \
  -d '{"query": "run rag on toy pdf and check if SMLP is published"}'
The agent receives the query,

Calls Ollama at localhost:11434,

Generates a task plan,

Executes it via your smlp_agent.py logic.

Terminal Setup Summary
Terminal     Purpose
1. Run ollama serve
2. Run uvicorn api_smlp_agent:app --reload
3. Send curl requests or use Swagger UI (http://127.0.0.1:8000/docs)

You can also use Swagger UI instead of curl.

Final Notes: 
--Yes, it's best to run each of these in its own shell/tab.
--Make sure your LLMInterpreter in smlp_agent.py is set to: LLMInterpreter(provider="ollama", model_name="mistral")

Let me know if you'd like:
'''