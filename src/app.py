import os
import sqlite3
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    redirect,
    url_for
)

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
    get_connection,
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
# Database helpers
# --------------------------------------------------

def row_value(
    row,
    column_name,
    position
):
    """
    Read a database row whether it comes
    from SQLite or PostgreSQL.
    """

    if isinstance(row, sqlite3.Row):

        return row[column_name]

    return row[position]


def update_conversation_title(
    conversation_id,
    user_id,
    title
):
    """
    Update a conversation title using
    either PostgreSQL or local SQLite.
    """

    connection = get_connection()

    cursor = connection.cursor()

    if os.getenv("DATABASE_URL"):

        cursor.execute(
            """
            UPDATE conversations
            SET title = %s
            WHERE id = %s
            AND user_id = %s
            """,
            (
                title,
                conversation_id,
                user_id
            )
        )

    else:

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


# --------------------------------------------------
# Security
# --------------------------------------------------

def request_is_same_origin():

    origin = request.headers.get(
        "Origin"
    )

    if origin:

        parsed_origin = urlparse(
            origin
        )

        request_origin = (
            f"{parsed_origin.scheme}://"
            f"{parsed_origin.netloc}"
        )

        expected_origin = (
            request.host_url.rstrip("/")
        )

        # Allow normal local development
        # whether Flask is opened through
        # localhost or 127.0.0.1.

        if (
            request_origin == expected_origin
        ):
            return True

        local_origins = {
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        }

        if (
            request_origin in local_origins
            and expected_origin in local_origins
        ):
            return True

        return False


    referer = request.headers.get(
        "Referer"
    )

    if referer:

        parsed_referer = urlparse(
            referer
        )

        referer_origin = (
            f"{parsed_referer.scheme}://"
            f"{parsed_referer.netloc}"
        )

        expected_origin = (
            request.host_url.rstrip("/")
        )

        if (
            referer_origin == expected_origin
        ):
            return True

        local_origins = {
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        }

        if (
            referer_origin in local_origins
            and expected_origin in local_origins
        ):
            return True

        return False


    # Local Flask development browsers
    # sometimes omit both Origin and Referer.
    #
    # We allow these requests locally.
    # Production HTTPS remains protected
    # by the origin/referer check.

    if request.host.startswith(
        (
            "127.0.0.1:",
            "localhost:"
        )
    ):

        return True

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
# Database initialization
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# OneTapSolve AI instructions
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are OneTapSolve AI, a professional, patient, and friendly IT support
technician.

Your purpose is to help users diagnose and resolve everyday technology
problems through safe, clear, interactive troubleshooting.

You can help with computers, Windows, macOS, Linux, Wi-Fi, internet,
networking, printers, scanners, software, email, browsers, accounts,
monitors, audio, cameras, Bluetooth, common hardware, basic IT security,
and workplace technology.

CORE SUPPORT PRINCIPLES:

1. Understand the user's problem and conversation context before responding.
2. Treat follow-up messages as part of the same troubleshooting conversation.
3. Use information the user has already provided.
4. Never ask the user to repeat information that is already available.
5. If the user says "I don't know," "I can't find it," "I'm not sure," or
   similar, do not simply ask what they mean. Explain exactly how they can
   find the information you previously requested.
6. If the user gives a vague word such as "it," "that," "this," or "the
   problem," determine what it most likely refers to from the immediately
   preceding conversation.
7. If the meaning is genuinely ambiguous, ask one short clarification
   question instead of guessing.
8. Remember short answers such as "yes," "no," "same," "still not working,"
   "it works," and "I don't know" in the context of the current problem.
9. Ask ONE useful diagnostic question at a time when a question is needed.
10. Give ONE primary troubleshooting action at a time whenever possible.
11. After the user reports the result, use that result to choose the next
    appropriate step.
12. Do not overwhelm beginners with a long list of unrelated solutions.

TROUBLESHOOTING METHOD:

1. Identify the user's goal and the symptoms.
2. Determine the most likely category of the problem.
3. Start with the safest and simplest diagnostic check.
4. Prefer diagnostic checks before configuration changes.
5. Prefer reversible solutions before advanced or disruptive changes.
6. Narrow down possible causes systematically.
7. Explain briefly what an important test result tells us when useful.
8. When enough evidence exists, state the most likely cause and why.
9. If a step fails, acknowledge that result and choose the next appropriate
   step instead of repeating the same instruction.
10. If the issue is solved, clearly say it appears resolved and explain what
    fixed it.
11. Always make the user's next action clear before ending the response.

BEGINNER-FRIENDLY SUPPORT:

Many users may not know technical terms or where to find information.

When you need information from a user, explain how to find it if they do not
know where it is.

For example, if you need a printer model:

- Tell the user to look at the front, top, side, or back of the printer.
- Explain that the model is usually printed near the printer name or logo.
- Give a simple example such as "HP LaserJet" or "Canon PIXMA."
- If appropriate, explain how to find the information from Windows or macOS.

If you need to know the operating system:

- For Windows, explain a simple way to identify the Windows version.
- For macOS, explain where to find the macOS version.
- For Linux, provide a simple identification method when appropriate.

If you need to know how a device is connected:

- Explain the difference between USB, Wi-Fi, Ethernet, and Bluetooth
  using simple language.
- Tell the user what physical cable or connection to look for.

Never assume a beginner knows where settings, device information, or
technical labels are located.

CONTEXT EXAMPLE:

If you ask:

"Could you tell me your printer model?"

and the user replies:

"I don't know how to find it."

Do NOT respond:

"I'm not sure what you're looking for."

Instead, understand that "it" refers to the printer model and explain how
to find the printer model step by step.

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
- Use plain language.
- Explain technical terms briefly when they are necessary.
- Keep responses concise and easy to follow.
- Use headings when useful.
- Number actions when there is more than one action.
- Give clear step-by-step instructions.
- Keep the conversation focused on the current IT problem.
- Do not overwhelm the user with a generic checklist.
- Do not repeat the same advice without explaining why.
- Do not blame the user for not knowing technical information.
- If the user is confused, simplify the explanation rather than repeating
  technical terminology.

DESIRED EXPERIENCE:

The user should feel like they are working with a real IT support technician.

For example, if the user says:

"My computer has no internet."

First determine whether the problem affects only that computer or the whole
network.

Ask whether another device on the same Wi-Fi can access the internet.

If another device works, focus on the affected computer.

If another device also fails, focus on the network, router, or internet
service.

Continue adapting each step based on the user's answers.

The goal is not simply to provide information. The goal is to guide the user
through the problem until it is solved or until the appropriate next level
of support is identified.
"""



# --------------------------------------------------
# AI response
# --------------------------------------------------

def get_ai_response(
    messages,
    provider_name
):

    if provider_name == "groq":
        response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tool_choice="none"
    )

        return (
            response
            .choices[0]
            .message
            .content
        )


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
# Sitemap
# --------------------------------------------------

@app.route(
    "/sitemap.xml"
)
def sitemap():

    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://fixflowitsupport.com/</loc>
    </url>
</urlset>
"""


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        username=session.get(
            "username"
        )
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


    session["user_id"] = row_value(
        user,
        "id",
        0
    )


    session["username"] = row_value(
        user,
        "username",
        1
    )


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
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


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
            session.get(
                "username"
            )
    })


# --------------------------------------------------
# Account page
# --------------------------------------------------

@app.route(
    "/account"
)
def account():

    if "user_id" not in session:

        return render_template(
            "login.html"
        )


    return render_template(
        "account.html",
        username=session.get(
            "username"
        )
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------
@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

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


    logged_in = "user_id" in session


    if not logged_in:

        guest_questions = session.get(
            "guest_questions",
            0
        )


        if guest_questions >= 3:

            return jsonify({
                "answer":
                    "You've used your 3 free questions. "
                    "Please create a free account or sign in "
                    "to continue using OneTapSolve AI.",
                "guest_limit_reached":
                    True
            }), 403


        session[
            "guest_questions"
        ] = guest_questions + 1


        provider_name = session.get(
            "provider",
            "groq"
        )


        try:

            messages = [
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_INSTRUCTIONS
                },

                {
                    "role":
                        "user",

                    "content":
                        user_message
                }
            ]


            answer = get_ai_response(
                messages,
                provider_name
            )


            remaining_questions = (
                3
                - session.get(
                    "guest_questions",
                    0
                )
            )


            return jsonify({
                "answer":
                    answer,

                "provider":
                    provider_name,

                "guest":
                    True,

                "remaining_questions":
                    remaining_questions
            })


        except Exception as error:

            print(
                "AI provider error:",
                error
            )


            session[
                "guest_questions"
            ] = max(
                0,
                session.get(
                    "guest_questions",
                    1
                ) - 1
            )


            return jsonify({
                "answer":
                    "Sorry, OneTapSolve AI could not connect "
                    "to the selected AI service.",

                "error":
                    str(error)
            }), 500


    user_id = session[
        "user_id"
    ]


    provider_name = session.get(
        "provider",
        "groq"
    )


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

        session[
            "conversation_id"
        ] = conversation_id


    previous_messages = get_messages(
        conversation_id,
        user_id
    )


    conversation = []


    for message in previous_messages:

        conversation.append({
            "role":
                row_value(
                    message,
                    "role",
                    0
                ),

            "content":
                row_value(
                    message,
                    "content",
                    1
                )
        })


    if len(previous_messages) == 0:

        title = user_message


        if len(title) > 45:

            title = (
                title[:45].rstrip()
                + "..."
            )


        update_conversation_title(
            conversation_id,
            user_id,
            title
        )


    conversation.append({
        "role":
            "user",

        "content":
            user_message
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
                provider_name,

            "conversation_id":
                conversation_id
        })


    except Exception as error:

        print(
            "AI provider error:",
            error
        )


        return jsonify({
            "answer":
                "Sorry, OneTapSolve AI could not connect "
                "to the selected AI service.",

            "error":
                str(error)
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


    user_id = session[
        "user_id"
    ]


    conversation_id = create_conversation(
        user_id,
        "New IT Support Session"
    )


    session[
        "conversation_id"
    ] = conversation_id


    return jsonify({
        "message":
            "Conversation reset.",

        "conversation_id":
            conversation_id
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


    user_id = session[
        "user_id"
    ]


    conversations = get_conversations(
        user_id
    )


    history_list = []


    for item in conversations:

        history_list.append({
            "id":
                row_value(
                    item,
                    "id",
                    0
                ),

            "title":
                row_value(
                    item,
                    "title",
                    1
                ),

            "created_at":
                row_value(
                    item,
                    "created_at",
                    2
                )
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


    user_id = session[
        "user_id"
    ]


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
                row_value(
                    message,
                    "role",
                    0
                ),

            "content":
                row_value(
                    message,
                    "content",
                    1
                ),

            "created_at":
                row_value(
                    message,
                    "created_at",
                    2
                )
        })


    session[
        "conversation_id"
    ] = history_id


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


    session[
        "provider"
    ] = provider_name


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