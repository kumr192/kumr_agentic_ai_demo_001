import json
import datetime as dt
import streamlit as st
from openai import OpenAI

# ---------------- Tools ----------------
def get_time() -> str:
    return dt.datetime.now().isoformat()

def calc(expression: str) -> str:
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

TOOL_MAP = {
    "get_time": lambda args: get_time(),
    "calc": lambda args: calc(args.get("expression", "")),
}

# ---------------- Agent loop ----------------
def run_agent(user_text: str, api_key: str):
    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You may call tools when useful. "
                "If a tool is required to answer, call it."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    tool_log = []

    # Allow a few tool calls max
    for _ in range(3):
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            tools=TOOLS,
        )

        tool_calls = [item for item in resp.output if item.type == "tool_call"]

        # If no tool calls, return the final text
        if not tool_calls:
            final_text = "".join(
                [item.text for item in resp.output if item.type == "output_text"]
            )
            return final_text.strip(), tool_log

        # Execute tools and append results
        for tc in tool_calls:
            name = tc.name
            raw_args = tc.arguments or "{}"

            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            tool_log.append({"tool": name, "arguments": args})

            if name not in TOOL_MAP:
                result = f"Unknown tool: {name}"
            else:
                result = TOOL_MAP[name](args)

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: too many tool calls.", tool_log


# ---------------- UI ----------------
st.set_page_config(page_title="BYOK Agentic Tool Demo", page_icon="🛠️", layout="centered")
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

user_text = st.text_input(
    "Your prompt", placeholder="Try: What time is it and compute (19*7)+3"
)

if st.button("Run"):
    api_key = (st.session_state.api_key or "").strip()

    if not api_key:
        st.error("Paste your OpenAI API key in the sidebar.")
    elif not user_text.strip():
        st.warning("Type something first.")
    else:
        try:
            answer, tool_log = run_agent(user_text, api_key)

            if tool_log:
                st.subheader("Tool calls")
                st.json(tool_log)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Runtime error: {e}")
