import os
import sqlite3
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from groq import Groq
from openai import OpenAI

from .database import (
    initialize_database,
    create_user,
    authenticate_user,
    create_conversation,
    add_message,
    get_messages,
    get_conversations,
    conversation_belongs_to_user,
)


# --------------------------------------------------
# Environment
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_FILE)


# --------------------------------------------------
# API keys
# --------------------------------------------------

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in .env"
    )

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found in .env"
    )


# --------------------------------------------------
# AI clients
# --------------------------------------------------

groq_client = Groq(
    api_key=groq_api_key
)

openai_client = OpenAI(
    api_key=openai_api_key
)


# --------------------------------------------------
# Flask
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY"
)

if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set."
    )

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    MAX_CONTENT_LENGTH=16 * 1024,
)


# --------------------------------------------------
# Security
# --------------------------------------------------

def request_is_same_origin():
    """
    Verify that browser state-changing requests
    originate from this application.
    """

    origin = request.headers.get("Origin")

    if origin:
        parsed_origin = urlparse(origin)

        request_origin = (
            f"{parsed_origin.scheme}://"
            f"{parsed_origin.netloc}"
        )

        expected_origin = (
            request.host_url.rstrip("/")
        )

        return request_origin == expected_origin

    referer = request.headers.get("Referer")

    if referer:
        parsed_referer = urlparse(referer)

        referer_origin = (
            f"{parsed_referer.scheme}://"
            f"{parsed_referer.netloc}"
        )

        expected_origin = (
            request.host_url.rstrip("/")
        )

        return referer_origin == expected_origin

    return False


@app.before_request
def protect_state_changing_requests():

    if request.method not in {
        "POST",
        "PUT",
        "PATCH",
        "DELETE"
    }:
        return None

    if not request_is_same_origin():

        return jsonify({
            "error":
                "Security check failed. "
                "Please refresh the page and try again."
        }), 403

    return None


# --------------------------------------------------
# Database
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# FixFlow-AI instructions
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are FixFlow-AI, a professional and friendly IT support technician.

Your purpose is to help users diagnose and resolve everyday technology
problems through safe, clear, interactive troubleshooting.

You can help with computers, Windows, macOS, Linux, Wi-Fi, internet,
networking, printers, scanners, software, email, browsers, accounts,
monitors, audio, cameras, Bluetooth, common hardware, basic IT security,
and workplace technology.

TROUBLESHOOTING METHOD:

1. Understand the user's problem before recommending changes.
2. Identify the most likely category and possible cause.
3. Start with the safest and simplest diagnostic check.
4. Ask ONE useful diagnostic question at a time when a question is needed.
5. Give ONE primary troubleshooting action at a time whenever possible.
6. After the user reports the result, use that result to choose the next
   question or action.
7. Interpret short answers such as "yes", "no", "same", "still not working",
   and "it works" using the conversation context.
8. Never repeatedly ask for information the user already supplied.
9. Explain briefly what an important test result tells us when useful.
10. Prefer diagnostic checks before configuration changes.
11. Prefer reversible solutions before advanced or disruptive changes.
12. If several causes are possible, narrow them down systematically.
13. When enough evidence exists, state the most likely cause and why.
14. If the issue is solved, clearly say it appears resolved and explain what
    fixed it.
15. If a step fails, acknowledge that result and choose the next appropriate
    diagnostic step instead of repeating the same instruction.
16. Make the user's next action clear before ending the response.

SAFETY:

- Never ask for passwords, API keys, authentication codes, security answers,
  full payment-card numbers, or other secrets.
- Never pretend you can see, access, control, or change the user's device,
  account, router, or network.
- Never claim you performed an action on the user's computer.
- Warn before steps that may restart a device, disconnect a connection,
  remove software, delete data, reset settings, or cause another important
  change.
- Before recommending a reset or destructive action, explain its consequence
  and prefer safer alternatives.
- For suspicious emails, links, downloads, malware, or account compromise,
  prioritize safety.
- Do not provide instructions intended to bypass security controls,
  authentication, or access restrictions.
- If professional repair, an administrator, a manufacturer, or an ISP is
  needed, say so clearly.

RESPONSE STYLE:

- Be professional, patient, encouraging, and friendly.
- Use plain language and explain technical terms briefly.
- Keep responses concise and easy to follow.
- Use headings when useful.
- Number actions when there is more than one action.
- Keep the conversation focused on the current IT problem.
- Do not overwhelm the user with a generic checklist.
- Do not repeat the same advice without explaining why.

DESIRED EXPERIENCE:

If the user says, "My computer has no internet," first determine whether
the problem affects only that computer or the whole network. For example,
ask whether another device on the same Wi-Fi can access the internet.

If another device works, focus on the affected computer.
If another device also fails, focus on the network, router, or service.

Continue adapting each step based on the user's answers so the experience
feels like working with a real IT support technician one step at a time.
"""


# --------------------------------------------------
# AI response
# --------------------------------------------------

def get_ai_response(messages, provider_name):

    if provider_name == "groq":

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages
        )

        return response.choices[0].message.content

    if provider_name == "openai":

        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=messages
        )

        return response.output_text

    raise ValueError(
        "Unknown AI provider"
    )


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.route("/")
def home():

    if "user_id" not in session:

        return render_template(
            "login.html"
        )

    return render_template(
        "index.html",
        username=session.get("username")
    )


# --------------------------------------------------
# Sign up
# --------------------------------------------------

@app.route(
    "/signup",
    methods=["POST"]
)
def signup():

    data = request.get_json()

    if not data:

        return jsonify({
            "error":
                "Please provide account information."
        }), 400

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    if not username or not password:

        return jsonify({
            "error":
                "Username and password are required."
        }), 400

    if len(username) < 3:

        return jsonify({
            "error":
                "Username must be at least 3 characters."
        }), 400

    if len(password) < 8:

        return jsonify({
            "error":
                "Password must be at least 8 characters."
        }), 400

    user_id = create_user(
        username,
        password
    )

    if user_id is None:

        return jsonify({
            "error":
                "That username already exists."
        }), 409

    session["user_id"] = user_id
    session["username"] = username

    session.pop(
        "conversation_id",
        None
    )

    session["provider"] = "groq"

    return jsonify({
        "message":
            "Account created successfully."
    })


# --------------------------------------------------
# Login
# --------------------------------------------------

@app.route(
    "/login",
    methods=["POST"]
)
def login():

    data = request.get_json()

    if not data:

        return jsonify({
            "error":
                "Please provide your login information."
        }), 400

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    user = authenticate_user(
        username,
        password
    )

    if user is None:

        return jsonify({
            "error":
                "Invalid username or password."
        }), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    session.pop(
        "conversation_id",
        None
    )

    session["provider"] = "groq"

    return jsonify({
        "message":
            "Login successful."
    })


# --------------------------------------------------
# Logout
# --------------------------------------------------

@app.route(
    "/logout",
    methods=["POST"]
)
def logout():

    session.clear()

    return jsonify({
        "message":
            "Logged out successfully."
    })


# --------------------------------------------------
# Current user
# --------------------------------------------------

@app.route(
    "/me",
    methods=["GET"]
)
def current_user():

    if "user_id" not in session:

        return jsonify({
            "logged_in": False
        })

    return jsonify({
        "logged_in": True,
        "username":
            session.get("username")
    })


# --------------------------------------------------
# Account page
# --------------------------------------------------

@app.route("/account")
def account():

    if "user_id" not in session:

        return render_template(
            "login.html"
        )

    return render_template(
        "account.html",
        username=session.get("username")
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    if "user_id" not in session:

        return jsonify({
            "answer":
                "Please log in first."
        }), 401

    user_id = session["user_id"]

    provider_name = session.get(
        "provider",
        "groq"
    )

    data = request.get_json()

    if not data or "message" not in data:

        return jsonify({
            "answer":
                "Please enter an IT problem."
        }), 400

    user_message = data[
        "message"
    ].strip()

    if not user_message:

        return jsonify({
            "answer":
                "Please describe your IT problem."
        }), 400

    if len(user_message) > 5000:

        return jsonify({
            "answer":
                "Please keep your IT problem under 5,000 characters."
        }), 400

    conversation_id = session.get(
        "conversation_id"
    )

    if conversation_id is not None:

        if not conversation_belongs_to_user(
            conversation_id,
            user_id
        ):

            conversation_id = None

    if conversation_id is None:

        conversation_id = create_conversation(
            user_id,
            "New IT Support Session"
        )

        session["conversation_id"] = (
            conversation_id
        )

    previous_messages = get_messages(
        conversation_id,
        user_id
    )

    conversation = []

    for message in previous_messages:

        conversation.append({
            "role":
                message["role"],
            "content":
                message["content"]
        })

    if len(previous_messages) == 0:

        title = user_message

        if len(title) > 45:

            title = (
                title[:45].rstrip()
                + "..."
            )

        connection = sqlite3.connect(
            os.path.join(
                BASE_DIR,
                "fixflow.db"
            )
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
            AND user_id = ?
            """,
            (
                title,
                conversation_id,
                user_id
            )
        )

        connection.commit()
        connection.close()

    conversation.append({
        "role": "user",
        "content": user_message
    })

    add_message(
        conversation_id,
        "user",
        user_message
    )

    try:

        messages = [
            {
                "role":
                    "system",
                "content":
                    SYSTEM_INSTRUCTIONS
            }
        ]

        messages.extend(
            conversation
        )

        answer = get_ai_response(
            messages,
            provider_name
        )

        add_message(
            conversation_id,
            "assistant",
            answer
        )

        return jsonify({
            "answer":
                answer,
            "provider":
                provider_name
        })

    except Exception as error:

        print(
            "AI provider error:",
            error
        )

        return jsonify({
            "answer":
                "Sorry, FixFlow-AI could not connect "
                "to the selected AI service."
        }), 500


# --------------------------------------------------
# New session
# --------------------------------------------------

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    if "user_id" not in session:

        return jsonify({
            "error":
                "Please log in first."
        }), 401

    user_id = session["user_id"]

    conversation_id = create_conversation(
        user_id,
        "New IT Support Session"
    )

    session["conversation_id"] = (
        conversation_id
    )

    return jsonify({
        "message":
            "Conversation reset."
    })


# --------------------------------------------------
# History
# --------------------------------------------------

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    if "user_id" not in session:

        return jsonify({
            "conversations": []
        })

    user_id = session["user_id"]

    conversations = get_conversations(
        user_id
    )

    history_list = []

    for item in conversations:

        history_list.append({
            "id":
                item["id"],
            "title":
                item["title"],
            "created_at":
                item["created_at"]
        })

    return jsonify({
        "conversations":
            history_list
    })


# --------------------------------------------------
# Previous conversation
# --------------------------------------------------

@app.route(
    "/history/<int:history_id>",
    methods=["GET"]
)
def history_messages(
    history_id
):

    if "user_id" not in session:

        return jsonify({
            "messages": []
        }), 401

    user_id = session["user_id"]

    if not conversation_belongs_to_user(
        history_id,
        user_id
    ):

        return jsonify({
            "error":
                "Conversation not found."
        }), 404

    messages = get_messages(
        history_id,
        user_id
    )

    message_list = []

    for message in messages:

        message_list.append({
            "role":
                message["role"],
            "content":
                message["content"],
            "created_at":
                message["created_at"]
        })

    session["conversation_id"] = (
        history_id
    )

    return jsonify({
        "messages":
            message_list
    })


# --------------------------------------------------
# Provider
# --------------------------------------------------

@app.route(
    "/provider",
    methods=["GET"]
)
def provider():

    if "user_id" not in session:

        return jsonify({
            "error":
                "Please log in first."
        }), 401

    provider_name = session.get(
        "provider",
        "groq"
    )

    return jsonify({
        "provider":
            provider_name
    })


@app.route(
    "/provider",
    methods=["POST"]
)
def change_provider():

    if "user_id" not in session:

        return jsonify({
            "error":
                "Please log in first."
        }), 401

    data = request.get_json()

    if not data or "provider" not in data:

        return jsonify({
            "error":
                "Provider was not specified."
        }), 400

    provider_name = str(
        data["provider"]
    ).lower().strip()

    if provider_name not in [
        "groq",
        "openai"
    ]:

        return jsonify({
            "error":
                "Unsupported AI provider."
        }), 400

    session["provider"] = provider_name

    return jsonify({
        "provider":
            provider_name
    })


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )