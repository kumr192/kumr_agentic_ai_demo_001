import json
import datetime as dt
import streamlit as st
from openai import OpenAI

# ---------------- Tools ----------------
def get_time() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def calc(expression: str) -> str:
    # Demo-only: tight character allowlist to reduce risk.
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "Rejected: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current server time in ISO format.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a basic arithmetic expression (+,-,*,/,% and parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(12*3)+4'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

def tool_execute(tool_name: str, args: dict) -> str:
    if tool_name == "get_time":
        return get_time()
    if tool_name == "calc":
        return calc(args.get("expression", ""))
    return f"Unknown tool: {tool_name}"

# ---------------- Agent loop (tool calling) ----------------
def run_agent(user_text: str, api_key: str):
    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use tools when they help. "
                "If you call a tool, do it explicitly."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    tool_log = []

    # up to 3 rounds of tool use
    for _ in range(3):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = resp.choices[0].message

        # No tool calls => final answer
        if not msg.tool_calls:
            return (msg.content or "").strip(), tool_log

        # IMPORTANT: append the assistant message that contains the tool calls
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )

        # Execute each tool call and append tool results
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            tool_log.append({"tool": tool_name, "arguments": args})

            result = tool_execute(tool_name, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    return "Stopped: too many tool calls.", tool_log

# ---------------- UI ----------------
st.set_page_config(page_title="Agentic Tool-Calling Demo (BYOK)", page_icon="🛠️", layout="centered")
st.title("Agentic Tool-Calling Demo (BYOK)")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.header("Bring Your Own Key")
    st.session_state.api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
    )
    if st.button("Clear key"):
        st.session_state.api_key = ""
        st.rerun()

    st.caption(
        "Each visitor must paste their own OpenAI API key. "
        "This server must receive your key to call OpenAI."
    )

user_text = st.text_input("Your prompt", placeholder="Try: What time is it and compute (19*7)+3")

if st.button("Run"):
    api_key = (st.session_state.api_key or "").strip()

    if not api_key:
        st.error("Paste your OpenAI API key in the sidebar.")
    elif not user_text.strip():
        st.warning("Type something first.")
    else:
        try:
            answer, tool_log = run_agent(user_text.strip(), api_key)

            if tool_log:
                st.subheader("Tool calls")
                st.json(tool_log)

            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Runtime error: {e}")
