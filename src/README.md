# Streamlit App Overview

## 📄 main.py

**Purpose:**  
Entry point and application router

**Responsibilities:**
- Page configuration  
- Session state management  
- Navigation logic  
- Main application flow control  

---

## 🔐 login.py

**Purpose:**  
Authentication and user management backend

**Responsibilities:**
- User credential verification  
- User registration  
- Password hashing (SHA-256)  
- User data persistence (JSON file storage)  
- User existence checking  

---

## 🎨 ui.py

**Purpose:**  
User interface components

**Responsibilities:**
- Login page rendering  
- Registration page rendering  
- Dashboard page rendering  
- Form validation  
- UI styling and layout  

---

## 🔧 Key Improvements Made

- **Security:** Added password hashing instead of storing plain text  
- **Validation:** Enhanced input validation (email format, password length, required fields)  
- **UI/UX:** Improved layout with columns, tabs, metrics, and better styling  
- **Data Storage:** Implemented JSON file-based user storage  
- **Modularity:** Clean separation of concerns between authentication, UI, and routing  

---

## ▶️ To Run the Application

1. Save all three files (`main.py`, `login.py`, `ui.py`) in the same directory.  
2. Run the app with:

   ```bash
   streamlit run main.py
