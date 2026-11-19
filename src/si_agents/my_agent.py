# %% [markdown]
#   ## Agent with memory
# 
#   ### Install all dependencies
# 
#   * Comment out once installed
# 
#   * uv add is an alternative to pip install.  Preferred due to maintaining python versions, etc.
# 
#   * The exclamation point allows it to run in Jupyter... makes it command to the underlying operating system terminal: run uv, and install this package
# 
#   * finish prompts and queries

# %%
# !uv add python-dotenv
# !uv add watchdog langchain-community langchain-chroma
# !uv add -U langchain-ollama
# !uv add langchain langchain-community langchain-chroma
# !uv add langchain-classic
# !uv add langchain
# !uv add sentence-transformers
# !uv add -qU langchain-chroma langchain-openai
# !uv add langchain langchain-ollama langgraph
# !uv add --upgrade jupyter ipywidgets


# %% [markdown]
#  ### Imports

# %%
import os
import asyncio

import json
import fnmatch
import chromadb

from dotenv import load_dotenv
from pathlib import Path
from ollama import AsyncClient
from ollama._utils import convert_function_to_tool
from chromadb.utils import embedding_functions
from memory_mgr import index_codebase, sub_query_knowledge_base, sub_save_to_knowledge_base
from fnmatch import fnmatch
from datetime import datetime
from IPython.display import display, Markdown



# %% [markdown]
# 

# %% [markdown]
# ### Hardcoding

# %%

# Globally track or pass your target codebase directory

REPORT_TO_WRITE = "Summarized_Business_Findings.md"
# REPORT_TO_WRITE = "Summarized_Findings.md"

# aka target_dir:  parameter representing the file path of the codebase folder the AI agent will analyze.
# or target_dirs (dirs) ... for multiple paths. Be sure to change main method header too.
CODE_TO_ANALYZE = "/home/choekstra/ipac/"
                  # "/home/choekstra/ipac/business/src/main/java/business"
                  # "/home/choekstra/ipac/business/src/main/java/business"
                  # "/home/choekstra/ipac/ipac.grails/grails-app"

PROJECT_ROOT = "/home/choekstra/ipac"
# PROJECT_ROOT = "/home/choekstra/ipac/ipac.grails/grails-app"

SYSTEM_PROMPT = """
Role: You are a Principal Database Architect and Legacy Modernization Expert specializing in complex JVM ecosystems. 
Your core expertise lies in reverse-engineering data layers that combine Spring Boot, Grails (GORM), Hibernate ORM, and Redis caching into a single monolithic architecture. 

Task: 
Start by calling sub_query_knowledge_base to search pre-indexed code and prior architectural notes before reading raw files. Use sub_save_to_knowledge_base to record major findings.
Before scanning files from scratch, check sub_query_knowledge_base to see if you already analyzed this component, and use sub_save_to_knowledge_base to store important findings. 
Analyze the provided codebase snippets, database schemas, or logs to map out the underlying SQL structure, identify framework-specific data patterns to help update sql grouping for website.
Call sub_query_knowledge_base at most 2–3 times. Once you have gathered enough context, STOP calling tools immediately and write your final architectural report.

[CONTEXT OVERVIEW]
When analyzing the code or schema, you must execute the following analytical steps:

1. **Schema Mapping & Conventions:** 
   - Identify whether tables are managed by GORM, Spring Data JPA, or external migration tools (Liquibase/Flyway).
   - Infer explicit vs. implicit table mappings, foreign keys, and join tables.

2. **Cache Invalidation & Redis Alignment:**
   - Map SQL write operations to their corresponding Redis cache evictions.
   - Highlight areas where direct SQL or native Hibernate queries can be edited, causing Redis cache desynchronization.

[INSTRUCTIONS FOR ANALYSIS EXECUTIONS]
Parse the code through the lens of a Grails 3.x + Spring Boot structural hierarchy.
NOTE: If this is a chunked file, analyze it as a partial snippet of a larger class context.
You are a strict tool-calling agent.
Rule 1: Never output text, explanations, or confirmations in the same turn as a tool call.
Rule 2: Call only one tool per turn.
Rule 3: If the knowledge base search returns no relevant results, do NOT attempt to call other tools or scan files. Simply state that the information is not in the knowledge base and ask the user for more context.


[OUTPUT FORMAT REQUIREMENT]
Structure your analysis using the following clear sections:
- **📊 Structural ERD Mapping:** An ASCII or Mermaid.js diagram illustrating the tables, relationships, and which framework controls them.
- **🔄 Redis & Cache Sync summary:** A breakdown of where the SQL database and the Redis cache are used.
- **🛠️ Actionable Recommendations:** Best process for changing SQL and Redis cache to make better SQL groupings of data. 

"""

USER_QUERY = """
Analyze how Redis is configured and used across this codebase. 
1. Query the knowledge base and scan for Redis components to identify connection setups, TTL policies, caching patterns, and pub/sub or queue usage.
2. Save a concise summary of your key architectural findings to the knowledge base using `sub_save_to_knowledge_base`.
3. Provide a final report covering:
   - **Current Architecture:** Where and how Redis is initialized and used.
   - **Risks & Bottlenecks:** Missing TTLs, connection pool issues, or single points of failure.
   - **Actionable Recommendations:** Concrete improvements with file references.
4. More detail on how Redis caching works with SQL Database.
5. Best practice for modifying SQL Database queries for grouped results.
6. As a PP using CPB, I want to analyze a list of resources grouped by Resource Function.
7. Currently we concatenate a species Resource Needs by Resource feature for indirect interaction in the CPB. We want to change this paradigm by concatenating by the Resource Function instead of the Resource Feature. Show me what code changes need to be made to accomplish that.
"""

EXCLUDE_DIRS = {'.git', 'node_modules', 'webpack', 'build', 'dist','.gitignore','.dockerignore','i18n',
                'build', 'buildspec', '.venv','.env', 'venv','ipac_storage_db','__pycache__','.github','.agents','taglib'
}
#                '**/scripts/**', '**/.gradle/**', 
#                '**/.git/**', '**/.gitignore/**', '**.dockerignore/**', '**/.gradle/**', 
#                '**/.idea/**', '**/.vscode/**', '**/node_modules/**', '**/.gsp/**', '**/assets/**', 
#                '**/domain/**', '**/i18n/**', '**/taglib/**', '**/.env/**', '**/.venv/**', "**/venv/**", 
#                "**/env/**", "**/__pycache__/**"'.github/**','**/.agents/**','**/ipac_storage/**'}
                
#                '**/.devcontainer/**', '**/bin/**', '**/lib/**', 
#                '**/build/**', '**/buildspec/**', '**/scripts/**', '**/.gradle/**', 
#                '**/.git/**', '**/.gitignore/**', '**.dockerignore/**', '**/.gradle/**', 
#                '**/.idea/**', '**/.vscode/**', '**/node_modules/**', '**/.gsp/**', '**/assets/**', 
#                '**/domain/**', '**/i18n/**', '**/taglib/**', '**/.env/**', '**/.venv/**', "**/venv/**", 
#                "**/env/**", "**/__pycache__/**"'.github/**','**/.agents/**','**/ipac_storage/**'}

#TARGET_EXTENSIONS = {'.groovy', '.java', '.yml', '.properties'}
#EXCLUDE_DIRS = {'build', '.gradle', '.git', 'node_modules', '.gsp'}

MAX_STEPS = 2000  # Protection loop ceiling - agentic AI using tools, in a loop


MAX_CHARS_PER_CHUNK = 6000 

REPORT_DESC = "Framework Matrix:** Grails 3.x / Spring Boot / GORM / Redis"
MODEL_IN_USE = os.getenv("MODEL_ENV")  

# Output format: 2026-10-27 08:42:15
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# TARGET_EXTENSIONS = {'.java'}
# TARGET_EXTENSIONS = {'**/*.*', '*.ipynb', '.md'}

DIRECTORY = {'*'}
PATTERN = {'**/*.*'}

MODEL_IN_USE = 'devstral-small-2:parser'



# %% [markdown]
# 

# %%
# Schemas telling Ollama what tools are available
# 1. Define the Tools Schema for Ollama

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Returns a list of all relevant source code files inside the workspace directory.",
            "arguments": {
                "type": "object", 
                "properties": {
                    "directory": {
                        "type": "string", 
                        "description": "The relative path to the file.",
                    },
                    "pattern": {
                        "type": "string", 
                        "description": "search pattern filter, eg: '*.groovy' or '*'"},
                    },
            },
            "required": ["directory", "pattern"]
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_chunk",
            "description": "Reads and returns a specific syntax-aware text chunk of a code file.",
            "properties": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "The relative path to the file."},
                    "chunk_index": {"type": "integer", "description": "The 1-based index of the chunk to read."}
                },
                "required": ["filepath", "chunk_index"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scan_codebase_for_redis",
            "description": "Statically scans the Spring Boot root path for files containing Redis annotations, configurations, or properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_root": {
                        "type": "string",
                        "description": "Absolute or relative path to the Spring Boot application directory."
                    }
                },
                "required": ["project_root"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_redis_component_file",
            "description": "Reads and analyzes a specific Spring Boot java or properties file to extract annotations, TTL configurations, or Serializers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the specific source file."
                    }
                },
                "required": ["file_path"]
            }
        }
    }
]                                  
                
CHROMA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "sub_query_knowledge_base",
            "description": "Search the ChromaDB vector database for relevant code snippets, configs, or architectural notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The search phrase or question to look up."
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of matching results to return (default 2)."
                    }
                },
                "required": ["query_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sub_save_to_knowledge_base",
            "description": "Save an architectural finding or summary into ChromaDB for future retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "A unique identifier string for this entry."
                    },
                    "summary_text": {
                        "type": "string",
                        "description": "The detailed text or architectural explanation to store."
                    },
                    "category": {
                        "type": "string",
                        "description": "Category label such as 'redis' or 'architecture'."
                    }
                },
                "required": ["doc_id", "summary_text"]
            }
        }
    }
]



# %% [markdown]
# ### Implement functions as Tools 

# %%

def holdlist_files(directory: str | Path, pattern: str) -> list[str]:
  """Scans the designated target directory for matching codebase files."""
  dir_path = Path(directory)
  return [str(p) for p in dir_path.glob(pattern) if p.is_file()]    


# passed in vars is a directory of type dir, and a pattern of type string:  
def list_files(directory: str, pattern: str) -> list[str]:
    """Scans the designated target directory for matching codebase files."""

    IGNORED_DIRS = EXCLUDE_DIRS

    print("in list_files")
    print(f"directory passed in: {directory}\n")
    print(f"pattern passed in: {pattern}\n")
    # print(f"Starting file scan in root directory: {CODE_TO_ANALYZE}")
    print(f"Starting file scan in root directory: {directory}\n")
    
    found_files = []
    base_path = Path(directory).resolve()
    print(f"base path 1: {base_path}\n")

    # Clean up the pattern in case the LLM wrapped it in extra wildcard stars
    # e.g., if pattern is "**/*.py", turn it into "*.py" to work safely with rglob
    clean_pattern = pattern.split("/")[-1] 

    found_files = []
    # Force clean recursive lookup
    for path in base_path.rglob(clean_pattern):
        if path.is_file() and not any(part in IGNORED_DIRS for part in path.parts):
            found_files.append(path)
            
    # return found_files
                
    print(f"Found {len(found_files)} relevant files in the codebase.\n")  

    return found_files

    # return json.dumps({"files": sorted(found_files)})

def split_code_file(code_content: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Syntax-aware file-splitter separating blocks at top-level brace boundaries."""
    
    print(f"Splitting code content into chunks with max {max_chars} characters each.")
    
    lines = code_content.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        current_chunk.append(line)
        current_length += len(line) + 1
        brace_count += stripped.count('{')
        brace_count -= stripped.count('}')
       
        if current_length >= max_chars and brace_count <= 0:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
            brace_count = 0
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def read_file_chunk(filepath: str, chunk_index: int) -> str:
    """Reads a chunk from a file relative to the codebase root directory."""
    
    print(f"Attempting to read chunk {chunk_index} from file: {filepath}")
    
    try:
        # Resolve the path safely against the codebase root
        base_path = Path(CODE_TO_ANALYZE).resolve()
        path = (base_path / filepath).resolve()
        
        # Security check: Ensure the agent isn't trying to escape the codebase directory
        if not path.is_relative_to(base_path):
            return json.dumps({"error": "Access denied: Path is outside the codebase root."})
            
        if not path.exists():
            return json.dumps({"error": f"File {filepath} not found."})
            
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        chunks = split_code_file(content)
        if chunk_index < 1 or chunk_index > len(chunks):
            return json.dumps({"error": f"Invalid chunk index. File has {len(chunks)} chunks."})
            
        return json.dumps({
            "filepath": filepath,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "content": chunks[chunk_index - 1]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def scan_codebase_for_redis(project_root: str) -> str:
    """Scan the codebase for Redis configurations, clients, and usage."""
    
    # Override '/' or '.' if devstral-small-2 passes it
    if not project_root or project_root.strip() in {"/", ".", "./"}:
        project_root = PROJECT_ROOT
    
    # Convert the updated string into a Path object so rglob uses the right folder!
    root_path = Path(project_root)
    print(f"Starting file scan in root directory: {root_path}")

    found_files = []
    markers = ["@Cacheable", "RedisTemplate", "RedisConnectionFactory", "@EnableCaching", "redis"]

    # Scan for common patterns in source directories
    for path in root_path.rglob("*.*"):
        try:
            content = path.read_text(errors="ignore")
            if any(marker in content for marker in ["@Cacheable", "RedisTemplate", "RedisConnectionFactory", "@EnableCaching"]):
                found_files.append(str(path))
        except Exception as e:
            continue
            
    if not found_files:
        return "Scan finished: No Redis components or application configurations detected."
        
    return f"Scan Results Found:\n" + "\n".join(f"- {f}" for f in found_files)


def parse_redis_component_file(file_path: str) -> str:
    """Reads specific file context to extract structural components."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File {file_path} not found."
    
    try:
        content = path.read_text(errors="ignore")
        # Extract annotations and references for the agent context matrix
        lines = content.splitlines()
        findings = [line.strip() for line in lines if any(x in line for x in ["@Cacheable", "RedisSerializer", "Duration", "TTL", "spring.redis"])]
        return f"File Content Analysis ({path.name}):\n" + "\n".join(findings)
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"



# %% [markdown]
# ### Convert Python functions, then add Chroma schemas for imported tools

# %%


existing_tools = [
     convert_function_to_tool(list_files),
     convert_function_to_tool(read_file_chunk),
     convert_function_to_tool(scan_codebase_for_redis),
     convert_function_to_tool(parse_redis_component_file),
]
all_tool_schemas = existing_tools + CHROMA_TOOLS_SCHEMA


# %% [markdown]
# ### map of tools used as needed in loops by the model. Agent inference relies on this.

# %%
TOOL_MAP = {
    "list_files": list_files,
    "read_file_chunk": read_file_chunk,
    "scan_codebase_for_redis": scan_codebase_for_redis,
    "parse_redis_component_file": parse_redis_component_file,
    "sub_query_knowledge_base": sub_query_knowledge_base,       # <-- ChromaDB semantic search
    "sub_save_to_knowledge_base": sub_save_to_knowledge_base,   # <-- ChromaDB persist memory
}


# %%
# import ollama
# !ollama show devstral-small-2:parser --modelfile | tail -n 45

# !ollama list
# print(
#     'Python client using ollama:', [m.model for m in ollama.list().models]
# )  
 

# %%
async def run_autonomous_agent(user_query: str, target_dir: str, model_name: str):
#async def run_autonomous_agent(user_query: str, target_dirs: str, model_name: str):


    # Pre-index the codebase before the agent starts thinking
    index_codebase(target_dir)

    client = AsyncClient()
    
    # Initialize message history with the user's starting prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    
    print("🦉 Initializing Autonomous Systems Architect Agent...\n")
    
    for step in range(MAX_STEPS):
        print(f"--- Step {step + 1} ---")
        cleaned_messages = []
        
        for msg in messages:
            # Safely convert Pydantic message objects to dicts if any sneak in
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
                
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = "\n".join(str(item) for item in content)
            elif content is None:
                msg["content"] = ""  # Ollama strictly prefers empty strings over None
                
            cleaned_messages.append(msg)

        response = await client.chat(
            model=model_name,
            messages=cleaned_messages,
            tools=all_tool_schemas,
            options={"num_ctx": 32768},  # <-- Prevents empty responses on long runs
            # options={"num_ctx": 16384},  # <-- Prevents empty responses on long runs
            stream=False
        )
        
        # Convert the pedantic response message to a dictionary before saving to history
        assistant_msg = response.message.model_dump()
        messages.append(assistant_msg)

        tool_calls = response.message.tool_calls or []

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = dict(tool_call.function.arguments)  # Dict in Ollama SDK

            # Force '/' or '.' to resolve to your actual project directory
            # otherwise, some models default to '/' as base root
            for key in ("project_root", "directory", "dir_path", "root_dir", "path"):
            
                if key in tool_args and str(tool_args[key]).strip() in {"", "/", ".", "./"}:
                    tool_args[key] = PROJECT_ROOT

                print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

                try:
                    # Execute the matching function and capture the output
                    if tool_name not in TOOL_MAP:
                        raise ValueError(f"Tool '{tool_name}' is not registered.")
                    
                        tool_result = TOOL_MAP[tool_name](**tool_args)
                        print(f"tool_result is: {tool_result}")
                    else:
                        tool_result = f"Error: Tool '{tool_name}' not found."
                        
                except Exception as e:
                    # 3. Catch the crash and feed the error back to the LLM as a tool response
                    tool_result = (
                    f"Error executing tool: {str(e)}. "
                    f"Available tools are: {list(TOOL_MAP.keys())}. "
                    "Do not invent tools. Answer using only the available tools or ask the user for help."
                )

                # Append the tool's output back into messages so the LLM can read it
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(tool_result) if not isinstance(tool_result, str) else tool_result,
                })
        else:
            # No tools were called, meaning the agent has reached its final answer

            content = response.message.content

            final_text = (content or "").strip()

            if not final_text:
                print("⚠️ Model returned an empty response. Nudging for final summary...")
                messages.append({
                    "role": "user",
                    "content": "You finished calling tools but returned an empty response. Write your complete final answer now."
                })
                continue

            print("\n✅ Final Response:\n")

            with open(REPORT_TO_WRITE, "w", encoding="utf-8") as file: file.write(content)
            
            display(Markdown(response.message.content))
            
            print(response.message.content)

            return response.message.content

    print("\n⚠️ Reached maximum steps without a final answer.")
    return None


if __name__ == "__main__":
    # Use asyncio.run() to execute top-level async functions in standard Python scripts outside of Jupyter:
    # asyncio.run(
    await run_autonomous_agent(
            user_query=USER_QUERY,
            target_dir=CODE_TO_ANALYZE,
            model_name=MODEL_IN_USE,
        )
    #)




