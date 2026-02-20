
import re
import hashlib
import os
from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError
import streamlit as st
from bson import ObjectId
from database import get_users_collection, get_portfolios_collection
# SECURITY: PASSWORD HASHING AND VERIFICATION 

# Salt for password hashing 
PASSWORD_SALT = os.getenv("PASSWORD_SALT")


def hash_password(password):

    # Combine password with salt
    salted_password = password + PASSWORD_SALT
    
    # Create SHA-256 hash
    hash_object = hashlib.sha256(salted_password.encode('utf-8'))
    
    # Return hexadecimal representation
    return hash_object.hexdigest()


def verify_password(plain_password, hashed_password):
    return hash_password(plain_password) == hashed_password


# SAFE COLLECTION ACCESS

def get_collection_safely(collection_getter):
    collection = collection_getter()
    if collection is None:
        return None, (False, "Database connection failed (collection unavailable)")
    return collection, None


# USER AUTHENTICATION

def verify_user(username, password):
    try:
        users = get_users_collection()
        if users is None:
            st.error("Database unavailable.")
            return False

        user = users.find_one({"username": username})
        if user:
            stored_hash = user.get("password", "")
            
            # Check if password is already hashed (64 char hex = SHA-256)
            if len(stored_hash) == 64:
                return verify_password(password, stored_hash)
            else:
                # Legacy: plain text password (migrate on successful login)
                if password == stored_hash:
                    # Migrate to hashed password
                    users.update_one(
                        {"username": username},
                        {"$set": {"password": hash_password(password)}}
                    )
                    return True
                return False

        return False

    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False


# USER VALIDATION UTILITIES

def user_exists(username):
    # Check if username already exists in database.

    try:
        users, error = get_collection_safely(get_users_collection)
        if error is not None:
            return True  # Assume exists

        return users.find_one({"username": username}) is not None

    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return True


def validate_email(email):
    #Validate email format using Regular Expression Module.

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_password_strength(password):
    #Check password meets security requirements.

    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    
    return True, "Password meets requirements"


# USER REGISTRATION

def register_user(username, password, email):
# Create a new user account.

    try:
        users, error = get_collection_safely(get_users_collection)
        if error is not None:
            return error

        # Validate Inputs
        if not username or not password or not email:
            return False, "All fields are required"

        if len(username) < 3:
            return False, "Username must be at least 3 characters"

        if user_exists(username):
            return False, "Username already exists"

        if not validate_email(email):
            return False, "Invalid email format"

        # Validate password strength
        is_valid, msg = validate_password_strength(password)
        if not is_valid:
            return False, msg

        if users.find_one({"email": email.lower()}):
            return False, "Email already registered"

        #Create User Document in collection
        user_doc = {
            "username": username,
            "password": hash_password(password),  # HASHED
            "email": email.lower(),
            "role": "user",
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "is_active": True,
        }

        users.insert_one(user_doc)
        return True, "Registration successful"

    except DuplicateKeyError:
        return False, "Username or email already exists"

    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False, "Registration failed"


#USER SESSION MANAGEMENT

def update_last_login(username):

    try:
        users, error = get_collection_safely(get_users_collection)
        if error is not None:
            return

        users.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )

    except Exception as e:
        st.error(f"Error updating login time: {str(e)}")


def get_user_info(username):

    try:
        users, error = get_collection_safely(get_users_collection)
        if error is not None:
            return None

        user = users.find_one({"username": username})

        if user:
            return {
                "username": user["username"],
                "email": user["email"],
                "role": user.get("role", "user"),
                "created_at": user.get("created_at"),
                "last_login": user.get("last_login"),
                "is_active": user.get("is_active", True),
            }

        return None

    except Exception as e:
        st.error(f"Error retrieving user info: {str(e)}")
        return None


# PASSWORD MANAGEMENT

def change_password(username, old_password, new_password):
    try:
        if not verify_user(username, old_password):
            return False, "Incorrect current password"

        # Validate new password strength
        is_valid, msg = validate_password_strength(new_password)
        if not is_valid:
            return False, msg

        users, error = get_collection_safely(get_users_collection)
        if error is not None:
            return error

        users.update_one(
            {"username": username},
            {"$set": {"password": hash_password(new_password)}},
        )

        return True, "Password changed successfully"

    except Exception as e:
        st.error(f"Error changing password: {str(e)}")
        return False, "Password change failed"


# PORTFOLIO MANAGEMENT

def create_portfolio(username, portfolio_data):

    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return error

        # Check for duplicate name
        existing = portfolios.find_one({
            "user_id": username,
            "portfolio_name": portfolio_data["name"],
        })
        if existing:
            return False, "Portfolio name already exists"

        portfolio_doc = {
            "user_id": username,
            "portfolio_name": portfolio_data["name"],
            "countries": portfolio_data["countries"],
            "stocks": portfolio_data.get("stocks", []),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_value": 0,
            "is_active": True,
        }

        result = portfolios.insert_one(portfolio_doc)
        return (True, "Portfolio created successfully") if result.inserted_id else (False, "Create failed")

    except Exception as e:
        return False, f"Error creating portfolio: {str(e)}"


def get_user_portfolios(username):
    """
    Get all active portfolios for a user.
    
    Args:
        username: Portfolio owner
        
    Returns:
        list: User's portfolios sorted by creation date
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return []

        return list(
            portfolios.find({"user_id": username, "is_active": True})
            .sort("created_at", -1)
        )

    except Exception as e:
        st.error(f"Error fetching portfolios: {str(e)}")
        return []


def get_all_portfolios():
    """
    Get all active portfolios (for community view).
    
    Returns:
        list: All active portfolios
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return []

        return list(
            portfolios.find({"is_active": True})
            .sort("created_at", -1)
        )

    except Exception as e:
        st.error(f"Error fetching all portfolios: {str(e)}")
        return []


def get_portfolio_by_id(portfolio_id):
    """
    Get a specific portfolio by ID.
    
    Args:
        portfolio_id: MongoDB ObjectId as string
        
    Returns:
        dict or None: Portfolio document
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return None

        return portfolios.find_one({"_id": ObjectId(portfolio_id)})

    except Exception as e:
        st.error(f"Error fetching portfolio: {str(e)}")
        return None


def update_portfolio(portfolio_id, update_data):
    """
    Update portfolio fields.
    
    Args:
        portfolio_id: Portfolio to update
        update_data: Fields to update
        
    Returns:
        tuple: (success, message)
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return error

        update_data["updated_at"] = datetime.now(timezone.utc)

        result = portfolios.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": update_data},
        )

        if result.modified_count > 0:
            return True, "Portfolio updated successfully"
        return False, "No changes made"

    except Exception as e:
        return False, f"Error updating portfolio: {str(e)}"


def delete_portfolio(portfolio_id, username):
    """
    Soft delete a portfolio (set is_active = False).
    
    Why Soft Delete?
    ---------------
    - Allows recovery of accidentally deleted data
    - Maintains referential integrity
    - Audit trail preservation
    
    Args:
        portfolio_id: Portfolio to delete
        username: Owner (for authorization)
        
    Returns:
        tuple: (success, message)
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return error

        result = portfolios.update_one(
            {"_id": ObjectId(portfolio_id), "user_id": username},
            {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
        )

        return (True, "Portfolio deleted") if result.modified_count > 0 else (False, "Portfolio not found")

    except Exception as e:
        return False, f"Error deleting portfolio: {str(e)}"


# ==== STOCK MANAGEMENT IN PORTFOLIOS ====

def add_stock_to_portfolio(portfolio_id, stock_data):
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return error

        # Check for duplicate stock
        portfolio = portfolios.find_one({"_id": ObjectId(portfolio_id)})
        if portfolio:
            for stock in portfolio.get("stocks", []):
                if stock["symbol"] == stock_data["symbol"]:
                    return False, f"Stock {stock_data['symbol']} already exists"

        result = portfolios.update_one(
            {"_id": ObjectId(portfolio_id)},
            {
                "$push": {"stocks": stock_data},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            },
        )

        return (True, f"Stock {stock_data['symbol']} added") if result.modified_count > 0 else (False, "Add failed")

    except Exception as e:
        return False, f"Error adding stock: {str(e)}"


def remove_stock_from_portfolio(portfolio_id, stock_symbol):
    """
    Remove a stock from a portfolio.
    
    Args:
        portfolio_id: Target portfolio
        stock_symbol: Symbol to remove
        
    Returns:
        tuple: (success, message)
    """
    try:
        portfolios, error = get_collection_safely(get_portfolios_collection)
        if error is not None:
            return error

        result = portfolios.update_one(
            {"_id": ObjectId(portfolio_id)},
            {
                "$pull": {"stocks": {"symbol": stock_symbol}},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            },
        )

        return (True, f"Stock {stock_symbol} removed") if result.modified_count > 0 else (False, "Stock not found")

    except Exception as e:
        return False, f"Error removing stock: {str(e)}"