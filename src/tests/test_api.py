# src/tests/test_api.py
"""Unit and integration tests for the ConvinceSense REST APIs."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app, get_db
from src.database.connection import Base
from src.database import models


# ── Test Database Setup ────────────────────────────────────────────────── #

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
# Using StaticPool keeps the same in-memory connection open across all TestClient requests
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the database dependency in the FastAPI application
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Create a clean database schema before each test."""
    Base.metadata.create_all(bind=engine)
    # Seed default user
    db = TestingSessionLocal()
    user = models.User(id=1, email="testrep@convincesense.com", name="Test Rep")
    db.add(user)
    db.commit()
    db.close()
    
    yield
    
    Base.metadata.drop_all(bind=engine)


# ── Tests ─────────────────────────────────────────────────────────────── #

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint():
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json()["framework"] == "ConvinceSense"


def test_create_and_get_calls():
    # 1. Create a Call
    call_payload = {
        "title": "Closing Deal Call",
        "user_id": 1,
        "records": [
            {
                "timestamp": 3.0,
                "score": 4,
                "transcript": "Yes, let's discuss pricing and sign the contract next week.",
                "sentiment": "POSITIVE",
                "buying_signals": ["pricing", "contract"],
                "hesitations": [],
                "detected_intents": ["PRICING", "COMMITMENT"],
                "intent_confidence": 0.9,
                "recommendation": "Prepare pricing breakdown",
                "energy": 0.85,
                "confidence": 0.95,
                "speaker": "Prospect"
            }
        ],
        "comments": [
            {"content": "Strong signals detected early."}
        ],
        "next_steps": [
            {"content": "Send proposal document", "completed": False}
        ]
    }
    response = client.post("/calls", json=call_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Closing Deal Call"
    assert data["user_id"] == 1
    assert len(data["records"]) == 1
    assert len(data["comments"]) == 1
    assert len(data["next_steps"]) == 1
    call_id = data["id"]

    # 2. Get list of calls
    list_response = client.get("/calls")
    assert list_response.status_code == 200
    calls_list = list_response.json()
    assert len(calls_list) == 1
    assert calls_list[0]["id"] == call_id

    # 3. Get single call details
    detail_response = client.get(f"/calls/{call_id}")
    assert detail_response.status_code == 200
    detail_data = detail_response.json()
    assert detail_data["title"] == "Closing Deal Call"
    assert detail_data["records"][0]["transcript"] == "Yes, let's discuss pricing and sign the contract next week."
    assert detail_data["comments"][0]["content"] == "Strong signals detected early."
    assert detail_data["next_steps"][0]["content"] == "Send proposal document"


def test_update_and_delete_call():
    # Setup call
    call_payload = {"title": "Discovery Call", "user_id": 1}
    response = client.post("/calls", json=call_payload)
    call_id = response.json()["id"]

    # 1. Update (toggle favorite)
    patch_response = client.patch(f"/calls/{call_id}", json={"is_favorite": True})
    assert patch_response.status_code == 200
    assert patch_response.json()["is_favorite"] is True

    # 2. Soft Delete
    delete_response = client.delete(f"/calls/{call_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True

    # 3. Verify it is not returned in GET /calls anymore
    list_response = client.get("/calls")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 0

    # 4. Verify detail returns 404
    detail_response = client.get(f"/calls/{call_id}")
    assert detail_response.status_code == 404


def test_call_actions():
    # Setup call
    call_payload = {
        "title": "Demo Call",
        "user_id": 1,
        "next_steps": [{"content": "Follow up email", "completed": False}]
    }
    response = client.post("/calls", json=call_payload)
    call_data = response.json()
    call_id = call_data["id"]
    next_step_id = call_data["next_steps"][0]["id"]

    # 1. Add comment
    comment_response = client.post(f"/calls/{call_id}/comments", json={"content": "Added feedback"})
    assert comment_response.status_code == 200
    assert comment_response.json()["content"] == "Added feedback"

    # 2. Update next step (mark completed)
    next_step_payload = {
        "id": next_step_id,
        "completed": True
    }
    step_response = client.patch(f"/calls/{call_id}/next-steps", json=next_step_payload)
    assert step_response.status_code == 200
    assert step_response.json()["completed"] is True


def test_analytics_and_bant():
    # 1. Create BANT-compliant call
    call_payload_1 = {
        "title": "Compliant Call",
        "user_id": 1,
        "records": [
            {
                "timestamp": 1.0,
                "score": 4,
                "transcript": "Let's talk about pricing. The decision maker is our VP who approved the budget.",
                "sentiment": "POSITIVE",
                "detected_intents": ["PRICING"]
            },
            {
                "timestamp": 2.0,
                "score": 4,
                "transcript": "Our need is to solve this problem by next week.",
                "sentiment": "NEUTRAL",
                "detected_intents": ["INFORMATION", "COMMITMENT"]
            }
        ]
    }
    client.post("/calls", json=call_payload_1)

    # 2. Create partially compliant call
    call_payload_2 = {
        "title": "Partial Call",
        "user_id": 1,
        "records": [
            {
                "timestamp": 1.0,
                "score": 3,
                "transcript": "I am looking for information, but we don't have a budget.",
                "sentiment": "NEUTRAL",
                "detected_intents": ["INFORMATION"]
            }
        ]
    }
    client.post("/calls", json=call_payload_2)

    # 3. Get Analytics
    analytics_response = client.get("/analytics")
    assert analytics_response.status_code == 200
    analytics = analytics_response.json()
    assert analytics["total_calls"] == 2
    assert analytics["average_interest_score"] == pytest.approx(3.67, 0.01)
    
    # Call 1 satisfies: Budget, Authority, Need, Timeline (4/4 = 1.0)
    # Call 2 satisfies: Need, Budget (2/4 = 0.5)
    # Expected average: (1.0 + 0.5) / 2 = 0.75
    assert analytics["bant_compliance_rate"] == pytest.approx(0.75, 0.01)
    assert analytics["bant_breakdown"]["budget_count"] == 2
    assert analytics["bant_breakdown"]["authority_count"] == 1
    assert analytics["bant_breakdown"]["need_count"] == 2
    assert analytics["bant_breakdown"]["timeline_count"] == 1
