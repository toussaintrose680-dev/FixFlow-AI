import os
import sqlite3
import hashlib
import secrets

DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_FILE = os.path.join(BASE_DIR, "fixflow.db")


def get_connection():
    if DATABASE_URL:
        import psycopg
        return psycopg.connect(DATABASE_URL)

    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                title TEXT,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
        """)

    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [column["name"] for column in columns]

        if "user_id" not in column_names:
            cursor.execute("""
                ALTER TABLE conversations
                ADD COLUMN user_id INTEGER
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
            )
        """)

        cursor.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
        first_user = cursor.fetchone()

        if first_user:
            cursor.execute("""
                UPDATE conversations
                SET user_id = ?
                WHERE user_id IS NULL
            """, (first_user["id"],))

    connection.commit()
    connection.close()


def hash_password(password):
    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return salt + ":" + password_hash


def verify_password(password, stored_password):
    try:
        salt, password_hash = stored_password.split(":")

        new_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        ).hex()

        return secrets.compare_digest(new_hash, password_hash)

    except ValueError:
        return False


def create_user(username, password):
    connection = get_connection()
    cursor = connection.cursor()

    password_hash = hash_password(password)

    try:
        if DATABASE_URL:
            cursor.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s)
                RETURNING id
                """,
                (username, password_hash)
            )

            user_id = cursor.fetchone()[0]

        else:
            cursor.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
                """,
                (username, password_hash)
            )

            user_id = cursor.lastrowid

        connection.commit()
        return user_id

    except sqlite3.IntegrityError:
        connection.rollback()
        return None

    except Exception:
        connection.rollback()

        if DATABASE_URL:
            try:
                cursor.execute(
                    "SELECT id FROM users WHERE username = %s",
                    (username,)
                )
            except Exception:
                connection.close()
                raise
        else:
            cursor.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,)
            )

        existing_user = cursor.fetchone()

        if existing_user:
            return None

        raise

    finally:
        connection.close()


def get_user(username):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            SELECT id, username, password_hash, created_at
            FROM users
            WHERE username = %s
            """,
            (username,)
        )
    else:
        cursor.execute(
            """
            SELECT id, username, password_hash, created_at
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

    user = cursor.fetchone()
    connection.close()

    return user


def authenticate_user(username, password):
    user = get_user(username)

    if not user:
        return None

    if isinstance(user, sqlite3.Row):
        stored_password = user["password_hash"]
    else:
        stored_password = user[2]

    if verify_password(password, stored_password):
        return user

    return None


def create_conversation(user_id, title="New IT Support Session"):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            INSERT INTO conversations (user_id, title)
            VALUES (%s, %s)
            RETURNING id
            """,
            (user_id, title)
        )

        conversation_id = cursor.fetchone()[0]

    else:
        cursor.execute(
            """
            INSERT INTO conversations (user_id, title)
            VALUES (?, ?)
            """,
            (user_id, title)
        )

        conversation_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return conversation_id


def add_message(conversation_id, role, content):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (conversation_id, role, content)
        )
    else:
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content)
        )

    connection.commit()
    connection.close()


def get_messages(conversation_id, user_id):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            SELECT messages.role, messages.content, messages.created_at
            FROM messages
            JOIN conversations
            ON messages.conversation_id = conversations.id
            WHERE messages.conversation_id = %s
            AND conversations.user_id = %s
            ORDER BY messages.id ASC
            """,
            (conversation_id, user_id)
        )
    else:
        cursor.execute(
            """
            SELECT messages.role, messages.content, messages.created_at
            FROM messages
            JOIN conversations
            ON messages.conversation_id = conversations.id
            WHERE messages.conversation_id = ?
            AND conversations.user_id = ?
            ORDER BY messages.id ASC
            """,
            (conversation_id, user_id)
        )

    messages = cursor.fetchall()
    connection.close()

    return messages


def get_conversations(user_id):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            SELECT id, title, created_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY id DESC
            """,
            (user_id,)
        )
    else:
        cursor.execute(
            """
            SELECT id, title, created_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,)
        )

    conversations = cursor.fetchall()
    connection.close()

    return conversations


def conversation_belongs_to_user(conversation_id, user_id):
    connection = get_connection()
    cursor = connection.cursor()

    if DATABASE_URL:
        cursor.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = %s
            AND user_id = %s
            """,
            (conversation_id, user_id)
        )
    else:
        cursor.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = ?
            AND user_id = ?
            """,
            (conversation_id, user_id)
        )

    result = cursor.fetchone()
    connection.close()

    return result is not None