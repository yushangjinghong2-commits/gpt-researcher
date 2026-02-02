import asyncio
import json
import os
import re
import time
import shutil
import traceback
from typing import Awaitable, Dict, List, Any
from fastapi.responses import JSONResponse, FileResponse
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher import GPTResearcher
from utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import logging

# Import chat agent
try:
    import sys
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from chat.chat import ChatAgentWithMemory
except ImportError:
    ChatAgentWithMemory = None

logger = logging.getLogger(__name__)

class CustomLogsHandler:
    """Custom handler to capture streaming logs from the research process"""
    def __init__(self, websocket, task: str):
        self.logs = []
        self.websocket = websocket
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
        self.log_file = os.path.join("outputs", f"{sanitized_filename}.json")
        self.timestamp = datetime.now().isoformat()
        # Initialize log file with metadata
        os.makedirs("outputs", exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "events": [],
                "content": {
                    "query": "",
                    "sources": [],
                    "context": [],
                    "report": "",
                    "costs": 0.0
                }
            }, f, indent=2)

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Store log data and send to websocket"""
        # Send to websocket for real-time display
        if self.websocket:
            await self.websocket.send_json(data)
            
        # Read current log file
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
            
        # Update appropriate section based on data type
        if data.get('type') == 'logs':
            log_data['events'].append({
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "data": data
            })
        else:
            # Update content section for other types of data
            log_data['content'].update(data)
            
        # Save updated log file
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def generate_task_graph(self) -> str:
        base = os.path.splitext(self.log_file)[0]
        graph_path = base + "_graph.md"
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
        query = ""
        if isinstance(log_data.get("content"), dict):
            query = log_data["content"].get("query", "")
        report_query = ""
        parent_query_hint = ""
        events = log_data.get("events", [])
        
        # Track tasks and their relationships
        tasks: Dict[str, Dict[str, Any]] = {}
        # current task for source mapping
        current: str | None = None
        
        # Helper to ensure task exists
        def get_task(name):
            if name not in tasks:
                tasks[name] = {"parent": None, "children": [], "sources": [], "tokens": 0, "elapsed_ms": 0}
            return tasks[name]

        # Initialize root task
        if query:
            get_task(query)

        for ev in events:
            data = ev.get("data", {})
            if data.get("type") != "logs":
                continue
            content = data.get("content")
            
            if content == "starting_research":
                out = data.get("output", "")
                meta = data.get("metadata") or {}
                # Try to get task name from metadata or parse from output
                task_name = meta.get("task")
                if not task_name:
                    m = re.search(r"Starting the research task for '(.+)'", out)
                    if m:
                        task_name = m.group(1)
                
                if task_name:
                    current = task_name
                    task_entry = get_task(task_name)
                    parent = meta.get("parent_query")
                    if parent:
                        task_entry["parent"] = parent
                        parent_entry = get_task(parent)
                        if task_name not in parent_entry["children"]:
                            parent_entry["children"].append(task_name)
                        if not parent_query_hint:
                            parent_query_hint = parent
                    elif query and task_name != query and not task_entry["parent"]:
                        # If no parent specified, assume it's a child of the main query (legacy fallback)
                        # But only if it's not the main query itself
                        task_entry["parent"] = query
                        if task_name not in get_task(query)["children"]:
                            get_task(query)["children"].append(task_name)
                    elif report_query and task_name != report_query and not task_entry["parent"]:
                        task_entry["parent"] = report_query
                        if task_name not in get_task(report_query)["children"]:
                            get_task(report_query)["children"].append(task_name)

            elif content == "subqueries":
                # Legacy handling for subqueries event
                meta = data.get("metadata") or []
                if isinstance(meta, list):
                    for sq in meta:
                        get_task(sq) # Ensure it exists
                        # Prefer current task as parent; fallback to main query if available
                        parent_name = current or query or report_query
                        if parent_name and sq != parent_name:
                            sq_entry = get_task(sq)
                            if not sq_entry["parent"]:
                                sq_entry["parent"] = parent_name
                                if sq not in get_task(parent_name)["children"]:
                                    get_task(parent_name)["children"].append(sq)

            elif content == "running_subquery_research":
                out = data.get("output", "")
                m = re.search(r"Running research for '(.+)'", out)
                if m:
                    current = m.group(1)
                    get_task(current)

            elif content == "fetching_query_content":
                out = data.get("output", "")
                m = re.search(r"Getting relevant content based on query: (.+?)\.\.\.", out)
                if m:
                    current = m.group(1)
                    get_task(current)

            elif content == "added_source_url":
                url = data.get("metadata")
                if current and url:
                    task = get_task(current)
                    if url not in task["sources"]:
                        task["sources"].append(url)

            elif content == "cost_update":
                meta = data.get("metadata")
                if current and isinstance(meta, dict):
                    pt = meta.get("prompt_tokens") or 0
                    ct = meta.get("completion_tokens") or 0
                    el = meta.get("elapsed_ms") or 0
                    task = get_task(current)
                    task["tokens"] += (pt + ct)
                    task["elapsed_ms"] += el
            
            elif content in ("writing_report", "report_written"):
                if not report_query:
                    out = data.get("output", "")
                    m = re.search(r"for '(.+?)'", out)
                    if m:
                        report_query = m.group(1)
                        get_task(report_query)
        
        # --- Tree Reconstruction Logic ---
        # 1. Collect all potential tasks
        all_task_names = set(tasks.keys())
        
        # 2. Identify root candidates (tasks with no parent)
        roots = [name for name in all_task_names if not tasks[name]["parent"]]
        
        # 3. Heuristic: If multiple roots exist, and one of them is the main query, 
        #    assume others are likely children of the main query if they are not explicitly disconnected.
        #    However, in deep research, subtasks SHOULD have parents.
        #    If they don't, it's a data gap. We attach them to Main Query for visualization.
        if not query and report_query:
            query = report_query
        if not query and parent_query_hint:
            query = parent_query_hint
        if not query and len(roots) == 1:
            query = roots[0]
        if not query:
            root_with_children = [name for name in roots if tasks.get(name, {}).get("children")]
            if len(root_with_children) == 1:
                query = root_with_children[0]
        if query and query in tasks:
            for root in roots:
                if root != query:
                    tasks[root]["parent"] = query
                    if root not in tasks[query]["children"]:
                        tasks[query]["children"].append(root)

        # Build Markdown Tree
        lines: List[str] = []
        lines.append(f"# 任务执行结构树")
        lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        def write_task_node(name, depth=0):
            indent = "  " * depth
            task = tasks.get(name)
            if not task:
                return
            
            # Task Title
            # Use different icons for different depths to visually distinguish
            icon = "🟢" if depth == 0 else ("🔵" if depth == 1 else "🟣")
            lines.append(f"{indent}- {icon} **{name}**")
            
            # Metrics
            metrics = [f"Tokens: {task['tokens']}"]
            if task["elapsed_ms"] > 0:
                metrics.append(f"Time: {task['elapsed_ms']}ms")
            lines.append(f"{indent}  - 📊 " + ", ".join(metrics))
            
            # Sources
            lines.append(f"{indent}  - 🔗 **来源**:")
            if task["sources"]:
                for u in task["sources"]:
                    lines.append(f"{indent}    - <{u}>")
            else:
                lines.append(f"{indent}    - 无")
            
            # Children - strictly nested
            if task["children"]:
                for child in task["children"]:
                    write_task_node(child, depth + 1)

        # Start with root(s)
        if query and query in tasks:
            write_task_node(query)
        else:
            # Fallback for when root query is not explicitly in tasks or tasks are disconnected
            for name, task in tasks.items():
                if not task["parent"]:
                    write_task_node(name)

        content_md = "\n".join(lines)
        with open(graph_path, "w", encoding="utf-8") as f:
            f.write(content_md)
        return graph_path


class Researcher:
    def __init__(self, query: str, report_type: str = "research_report"):
        self.query = query
        self.report_type = report_type
        # Generate unique ID for this research task
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
        # Initialize logs handler with research ID
        self.logs_handler = CustomLogsHandler(None, self.research_id)
        self.researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            websocket=self.logs_handler
        )

    async def research(self) -> dict:
        """Conduct research and return paths to generated files"""
        await self.researcher.conduct_research()
        report = await self.researcher.write_report()
        
        # Generate the files
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)
        
        # Get the JSON log path that was created by CustomLogsHandler
        json_relative_path = os.path.relpath(self.logs_handler.log_file)
        
        return {
            "output": {
                **file_paths,  # Include PDF, DOCX, and MD paths
                "json": json_relative_path
            }
        }

def sanitize_filename(filename: str) -> str:
    # Split into components
    prefix, timestamp, *task_parts = filename.split('_')
    task = '_'.join(task_parts)
    
    # Calculate max length for task portion
    # 255 - len(os.getcwd()) - len("\\gpt-researcher\\outputs\\") - len("task_") - len(timestamp) - len("_.json") - safety_margin
    max_task_length = 255 - len(os.getcwd()) - 24 - 5 - 10 - 6 - 5  # ~189 chars for task
    
    # Truncate task if needed
    truncated_task = task[:max_task_length] if len(task) > max_task_length else task
    
    # Reassemble and clean the filename
    sanitized = f"{prefix}_{timestamp}_{truncated_task}"
    return re.sub(r"[^\w\s-]", "", sanitized).strip()


async def handle_start_command(websocket, data: str, manager):
    json_data = json.loads(data[6:])
    (
        task,
        report_type,
        source_urls,
        document_urls,
        tone,
        headers,
        report_source,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    ) = extract_command_data(json_data)

    if not task or not report_type:
        print("Error: Missing task or report_type")
        return

    # Create logs handler with websocket and task
    logs_handler = CustomLogsHandler(websocket, task)
    # Initialize log content with query
    await logs_handler.send_json({
        "query": task,
        "sources": [],
        "context": [],
        "report": ""
    })

    sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")

    report = await manager.start_streaming(
        task,
        report_type,
        report_source,
        source_urls,
        document_urls,
        tone,
        websocket,
        headers,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    )
    report = str(report)
    file_paths = await generate_report_files(report, sanitized_filename)
    # Add JSON log path to file_paths
    file_paths["json"] = os.path.relpath(logs_handler.log_file)
    graph_path = logs_handler.generate_task_graph()
    file_paths["task_graph_md"] = os.path.relpath(graph_path)
    await send_file_paths(websocket, file_paths)


async def handle_human_feedback(data: str):
    feedback_data = json.loads(data[14:])  # Remove "human_feedback" prefix
    print(f"Received human feedback: {feedback_data}")
    # TODO: Add logic to forward the feedback to the appropriate agent or update the research state


async def handle_chat_command(websocket, data: str):
    """Handle chat command from WebSocket."""
    try:
        # Parse chat data - format is "chat {json_data}"
        json_str = data[5:].strip()  # Remove "chat " prefix
        chat_data = json.loads(json_str)
        
        message = chat_data.get("message", "")
        report = chat_data.get("report", "")
        messages = chat_data.get("messages", [])
        
        # If only message is provided, convert to messages format
        if message and not messages:
            messages = [{"role": "user", "content": message}]
        
        if not messages:
            await websocket.send_json({
                "type": "chat",
                "content": "No message provided.",
                "role": "assistant"
            })
            return
        
        # Check if ChatAgentWithMemory is available
        if ChatAgentWithMemory is None:
            await websocket.send_json({
                "type": "chat",
                "content": "Chat functionality is not available. Please check the server configuration.",
                "role": "assistant"
            })
            return
        
        # Create chat agent with the report context
        chat_agent = ChatAgentWithMemory(
            report=report,
            config_path="default",
            headers=None
        )
        
        # Process the chat
        response_content, tool_calls_metadata = await chat_agent.chat(messages, websocket)
        
        # Send response back via WebSocket
        await websocket.send_json({
            "type": "chat",
            "content": response_content,
            "role": "assistant",
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        })
        
        logger.info(f"Chat response sent successfully")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse chat data: {e}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error: Invalid message format - {str(e)}",
            "role": "assistant"
        })
    except Exception as e:
        logger.error(f"Error handling chat command: {e}\n{traceback.format_exc()}")
        await websocket.send_json({
            "type": "chat",
            "content": f"Error processing your message: {str(e)}",
            "role": "assistant"
        })

async def generate_report_files(report: str, filename: str) -> Dict[str, str]:
    pdf_path = await write_md_to_pdf(report, filename)
    docx_path = await write_md_to_word(report, filename)
    md_path = await write_text_to_md(report, filename)
    return {"pdf": pdf_path, "docx": docx_path, "md": md_path}


async def send_file_paths(websocket, file_paths: Dict[str, str]):
    await websocket.send_json({"type": "path", "output": file_paths})


def get_config_dict(
    langchain_api_key: str, openai_api_key: str, tavily_api_key: str,
    google_api_key: str, google_cx_key: str, bing_api_key: str,
    searchapi_api_key: str, serpapi_api_key: str, serper_api_key: str, searx_url: str
) -> Dict[str, str]:
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "")
    }


def update_environment_variables(config: Dict[str, str]):
    for key, value in config.items():
        os.environ[key] = value


async def handle_file_upload(file, DOC_PATH: str) -> Dict[str, str]:
    file_path = os.path.join(DOC_PATH, os.path.basename(file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File uploaded to {file_path}")

    document_loader = DocumentLoader(DOC_PATH)
    await document_loader.load()

    return {"filename": file.filename, "path": file_path}


async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
    file_path = os.path.join(DOC_PATH, os.path.basename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File deleted: {file_path}")
        return JSONResponse(content={"message": "File deleted successfully"})
    else:
        print(f"File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "File not found"})


async def execute_multi_agents(manager) -> Any:
    websocket = manager.active_connections[0] if manager.active_connections else None
    if websocket:
        report = await run_research_task("Is AI in a hype cycle?", websocket, stream_output)
        return {"report": report}
    else:
        return JSONResponse(status_code=400, content={"message": "No active WebSocket connection"})


async def handle_websocket_communication(websocket, manager):
    running_task: asyncio.Task | None = None

    def run_long_running_task(awaitable: Awaitable) -> asyncio.Task:
        async def safe_run():
            try:
                await awaitable
            except asyncio.CancelledError:
                logger.info("Task cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error running task: {e}\n{traceback.format_exc()}")
                await websocket.send_json(
                    {
                        "type": "logs",
                        "content": "error",
                        "output": f"Error: {e}",
                    }
                )

        return asyncio.create_task(safe_run())

    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data[:50]}..." if len(data) > 50 else data)
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif running_task and not running_task.done():
                    # discard any new request if a task is already running
                    logger.warning(
                        f"Received request while task is already running. Request data preview: {data[: min(20, len(data))]}..."
                    )
                    await websocket.send_json(
                        {
                            "type": "logs",
                            "content": "warning",
                            "output": "Task already running. Please wait.",
                        }
                    )
                # Normalize command detection by checking startswith after stripping whitespace
                elif data.strip().startswith("start"):
                    logger.info(f"Processing start command")
                    running_task = run_long_running_task(
                        handle_start_command(websocket, data, manager)
                    )
                elif data.strip().startswith("human_feedback"):
                    logger.info(f"Processing human_feedback command")
                    running_task = run_long_running_task(handle_human_feedback(data))
                elif data.strip().startswith("chat"):
                    logger.info(f"Processing chat command")
                    running_task = run_long_running_task(handle_chat_command(websocket, data))
                else:
                    error_msg = f"Error: Unknown command or not enough parameters provided. Received: '{data[:100]}...'" if len(data) > 100 else f"Error: Unknown command or not enough parameters provided. Received: '{data}'"
                    logger.error(error_msg)
                    print(error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "content": "error",
                        "output": "Unknown command received by server"
                    })
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}\n{traceback.format_exc()}")
                print(f"WebSocket error: {e}")
                break
    finally:
        if running_task and not running_task.done():
            running_task.cancel()

def extract_command_data(json_data: Dict) -> tuple:
    return (
        json_data.get("task"),
        json_data.get("report_type"),
        json_data.get("source_urls"),
        json_data.get("document_urls"),
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source"),
        json_data.get("query_domains", []),
        json_data.get("mcp_enabled", False),
        json_data.get("mcp_strategy", "fast"),
        json_data.get("mcp_configs", []),
    )
