import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    #MongoDB configuration, connection handling, health checks, and index creation.

    def __init__(self):
        self._client = None
        self._db = None
        self.database_name = os.getenv("DATABASE_NAME", "IA")

    # CONNECTION STRING
    @property
    def connection_string(self):
        #Return MongoDB URI from environment, checking several variable names.

        # Priority list of possible variable names
        possible_keys = [
            "MONGO_URI",
            "MONGODB_URI",
            "DATABASE_URL",
            "MONGODB_CONNECTION_STRING",
            "DB_URI",
        ]

        uri = None
        for key in possible_keys:
            uri = os.getenv(key)
            if uri:
                break

        if not uri:
            st.error(
                "MongoDB connection string not found. "
                "Set one of: MONGO_URI, MONGODB_URI, DATABASE_URL, "
                "MONGODB_CONNECTION_STRING, or DB_URI."
            )
            return None

        # Mask credentials in logs (but NOT returned)
        try:
            if "://" in uri and "@" in uri:
                before_at = uri.split("://")[1].split("@")[0]
                safe = uri.replace(before_at, "***:***")
            else:
                safe = "***"
            print(f"[INFO] MongoDB connection string loaded: {safe}")
        except Exception:
            pass

        return uri

    # ---- CONNECT ---
    def connect(self):
        #Establish database connection and return success boolean.
        if self._client is not None:
            return True  # Already connected

        uri = self.connection_string
        if not uri:
            return False

        try:
            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=20000,
                socketTimeoutMS=20000,
                maxPoolSize=10,
            )

            # Will raise if unreachable
            self._client.server_info()

            self._db = self._client[self.database_name]
            return True

        except ConnectionFailure as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            self._client = None
            return False

        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            self._client = None
            return False

    #Disconnect
    def disconnect(self):
        #Close active MongoDB client.
        if self._client:
            try:
                self._client.close()
            finally:
                self._client = None
                self._db = None

    # Acces to database & collection
    def get_database(self):
        #Return database instance, or None if unavailable.
        return self._db if self.connect() else None

    def get_collection(self, name: str):
        #Return collection reference, or None.
        db = self.get_database()
        return db[name] if db is not None else None

    #Health Check
    def health_check(self):
        try:
            if self._client is not None:
                self._client.admin.command("ping")
                return True
        except Exception as e:
            st.error(f"Database health check failed: {str(e)}")
        return False

    # creating indexes
    def create_indexes(self):
        #creating indexes for all collections
        db = self.get_database()
        if db is None:
            return False

        try:
            # Users
            users = db["users"]
            users.create_index("username", unique=True)
            users.create_index("email", unique=True)
            users.create_index("created_at")
            users.create_index("last_login")

            # Dashboard data
            dash = db["dashboard_data"]
            dash.create_index("user_id")
            dash.create_index("created_at")

            # Portfolios
            portfolios = db["portfolios"]
            portfolios.create_index("user_id")
            portfolios.create_index("created_at")
            portfolios.create_index("portfolio_name")

            print("[INFO] MongoDB indexes created successfully.")
            return True

        except Exception as e:
            st.warning(f"Could not create indexes: {str(e)}")
            return False


#GLOBAL INSTANCE 
db_config = DatabaseConfig()


#BACKWARD-COMPATIBLE HELPERS
def get_db():
    return db_config.get_database()


def get_users_collection():
    return db_config.get_collection("users")


def get_dashboard_collection():
    return db_config.get_collection("dashboard_data")


def get_portfolios_collection():
    return db_config.get_collection("portfolios")


def initialize_database():
    """Initialize DB connection + create indexes."""
    if db_config.connect():
        db_config.create_indexes()
        return True
    return False