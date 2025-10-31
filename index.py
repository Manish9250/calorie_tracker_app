import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from datetime import date
from typing import List

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file.")
genai.configure(api_key=GEMINI_API_KEY)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Please set it in Vercel.")

# --- Database Setup (SQLAlchemy) ---
Base = declarative_base()
engine = create_engine(DATABASE_URL)  
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    food_items = relationship("FoodItem", back_populates="owner")
    food_logs = relationship("FoodLog", back_populates="owner")

class FoodItem(Base):
    __tablename__ = "food_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)
    fiber = Column(Float)
    serving_size_unit = Column(String, default="unit") # e.g., '100g', '100ml', 'slice'
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="food_items")

class FoodLog(Base):
    __tablename__ = "food_logs"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    meal_type = Column(String)
    quantity = Column(Float, default=1.0)
    user_id = Column(Integer, ForeignKey("users.id"))
    food_item_id = Column(Integer, ForeignKey("food_items.id"))
    owner = relationship("User", back_populates="food_logs")
    food_item = relationship("FoodItem")

Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class FoodItemBase(BaseModel):
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    serving_size_unit: str

class FoodItemInDB(FoodItemBase):
    id: int
    user_id: int
    class Config:
        orm_mode = True

class FoodLogCreate(BaseModel):
    date: date
    meal_type: str
    quantity: float
    food_item_id: int

class FoodLogInDB(BaseModel):
    id: int
    date: date
    meal_type: str
    quantity: float
    food_item: FoodItemInDB
    class Config:
        orm_mode = True

class LoginRequest(BaseModel):
    username: str
    password: str

class FoodAnalysisRequest(BaseModel):
    description: str
    username: str

# --- FastAPI App ---
app = FastAPI()

def get_db():
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

# --- API Endpoints ---
@app.post("/api/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        user = User(username=request.username, password=request.password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"status": "success", "message": "User created."}
    if user.password != request.password:
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {"status": "success", "message": "Login successful."}

@app.post("/api/analyze-and-find-food")
def analyze_and_find_food(request: FoodAnalysisRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # IMPROVED PROMPT for structured parsing
    parsing_prompt = f"""
        Analyze the food description: "{request.description}".
        Extract the quantity, the unit of measurement, and the standardized item name.
        Return a single JSON object with three keys: "quantity", "unit", and "item_name".
        - "quantity" should be a number.
        - "unit" should be a string like 'ml', 'g', 'slice', 'piece', 'bowl', 'cup', or 'unit' if not specified.
        - "item_name" should be the food's name, singular and standardized (e.g., "boiled egg" instead of "eggs").

        Example 1: "500ml of milk" -> {{"quantity": 500, "unit": "ml", "item_name": "milk"}}
        Example 2: "2 slices of pepperoni pizza" -> {{"quantity": 2, "unit": "slice", "item_name": "pepperoni pizza"}}
        Example 3: "An apple" -> {{"quantity": 1, "unit": "unit", "item_name": "apple"}}
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(parsing_prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        parsed_result = json.loads(cleaned_response)
        
        quantity = parsed_result.get("quantity", 1.0)
        unit = parsed_result.get("unit", "unit").lower()
        item_name_raw = parsed_result.get("item_name", "").lower()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {e}")

    # NEW LOGIC: Normalize quantity based on unit
    normalized_quantity = float(quantity)
    serving_size_unit = unit
    
    if unit in ['ml', 'g', 'gram', 'grams']:
        normalized_quantity = float(quantity) / 100.0
        serving_size_unit = f"100{unit.replace('gram','g').replace('s','')}"

    existing_item = db.query(FoodItem).filter(FoodItem.user_id == user.id, FoodItem.name == item_name_raw).first()
    
    if existing_item:
        return {"item": existing_item, "quantity": normalized_quantity, "new": False}

    nutrition_prompt = f"""
        Provide estimated nutritional information for ONE serving of "{item_name_raw}".
        If the unit is 'g' or 'ml', provide data for a 100g or 100ml serving.
        Otherwise, provide data for one piece/slice/unit.
        Return a clean JSON object with keys: "name", "calories", "protein", "carbs", "fat", "fiber".
        The name should be a cleaned-up version. All values must be numbers.

    """
    try:
        response = model.generate_content(nutrition_prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        nutrition_data = json.loads(cleaned_response)
        
        new_item = FoodItem(
            name=nutrition_data.get("name", item_name_raw),
            calories=nutrition_data.get("calories", 0),
            protein=nutrition_data.get("protein", 0),
            carbs=nutrition_data.get("carbs", 0),
            fat=nutrition_data.get("fat", 0),
            fiber=nutrition_data.get("fiber", 0),
            serving_size_unit=serving_size_unit,
            user_id=user.id
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        return {"item": new_item, "quantity": normalized_quantity, "new": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI nutrition fetch failed: {e}")

@app.get("/api/food-items/{username}", response_model=List[FoodItemInDB])
def get_user_food_items(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.query(FoodItem).filter(FoodItem.user_id == user.id).order_by(FoodItem.name).all()

@app.post("/api/log/{username}", response_model=FoodLogInDB)
def create_food_log(username: str, log: FoodLogCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db_log = FoodLog(date=log.date, meal_type=log.meal_type, quantity=log.quantity, food_item_id=log.food_item_id, user_id=user.id)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@app.get("/api/log/{username}/{log_date}", response_model=List[FoodLogInDB])
def get_logs_for_date(username: str, log_date: date, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.query(FoodLog).filter(FoodLog.user_id == user.id, FoodLog.date == log_date).all()

@app.delete("/api/log/{log_id}")
def delete_log(log_id: int, db: Session = Depends(get_db)):
    log_to_delete = db.query(FoodLog).filter(FoodLog.id == log_id).first()
    if not log_to_delete:
        raise HTTPException(status_code=404, detail="Log not found")
    db.delete(log_to_delete)
    db.commit()
    return {"status": "success", "message": "Log deleted"}

@app.get("/api/stats/{username}/calendar-data")
def get_calendar_data(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        calories_per_day = (
            db.query(
                # --- CHANGE 1 ---
                # Query for the column directly, not func.date()
                FoodLog.date.label("log_date"),
                func.sum(FoodLog.quantity * FoodItem.calories).label("total_calories")
            )
            .select_from(FoodLog)
            .join(FoodItem, FoodLog.food_item_id == FoodItem.id)
            .filter(FoodLog.user_id == user.id)
            # --- CHANGE 2 ---
            # Group by the column directly
            .group_by(FoodLog.date)
            .all()
        )
        
        # Now, log_date will be a Python 'date' object,
        # so .isoformat() will work correctly.
        calendar_data = {
            log_date.isoformat(): total_calories
            for log_date, total_calories in calories_per_day
            if total_calories is not None
        }
        print(calendar_data)
        return calendar_data
        
    except Exception as e:
        print(f"Error fetching calendar data: {e}") 
        raise HTTPException(status_code=500, detail=f"Error fetching calendar data: {e}")
    

# --- ADD THIS FUNCTION BACK AT THE END ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    # Construct the path to index.html relative to this script's location
    # __file__ is the path to the current script (api/index.py)
    # os.path.dirname(__file__) is the directory (api/)
    # os.path.join(..., '..', 'index.html') goes up one level and finds index.html
    html_file_path = os.path.join(os.path.dirname(__file__), '.', 'index.html')
    
    try:
        with open(html_file_path, 'r') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        print(f"Error: index.html not found at expected path: {html_file_path}")
        raise HTTPException(status_code=404, detail="index.html not found.")
    except Exception as e:
        print(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal server error reading homepage.")