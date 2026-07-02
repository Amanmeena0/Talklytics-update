# src/tests/test_auth.py
"""Unit and integration tests for security, JWT, and authentication APIs."""

from datetime import timedelta
import sys
from pathlib import Path

from fastapi import Depends, status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app as main_app, get_db
from src.auth.dependencies import get_current_user, require_permission
from src.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.database import models
from src.database.connection import Base
from src.database.seed import seed_roles_and_permissions


# ── Test Database Setup ────────────────────────────────────────────────── #

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(main_app)


# Create a test-only protected route for RBAC testing (avoid name starting with 'test_')
@main_app.get("/dummy-protected-route")
def dummy_protected_endpoint(
    current_user=Depends(require_permission("calls:write")),
):
    return {"message": "Success", "user_id": current_user.id}


@pytest.fixture(autouse=True)
def setup_database():
    """Create a clean database schema and seed data before each test."""
    # Set override dynamically to prevent interference
    main_app.dependency_overrides[get_db] = override_get_db
    client.cookies.clear()
    
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    # Seed default roles and permissions
    seed_roles_and_permissions(db)

    db.close()
    yield
    Base.metadata.drop_all(bind=engine)
    
    # Clean up override
    main_app.dependency_overrides.pop(get_db, None)


# ── Password Hashing Tests ─────────────────────────────────────────────── #

def test_password_hashing():
    """Test that password hashing and verification works correctly."""
    pwd = "secretpassword123"
    hashed = hash_password(pwd)

    # Check verification works
    assert verify_password(hashed, pwd) is True

    # Check mismatch returns False
    assert verify_password(hashed, "wrongpassword") is False


# ── JWT Tests ─────────────────────────────────────────────────────────── #

def test_create_and_decode_jwt():
    """Test token encoding and decoding and expiration handling."""
    claims = {"sub": "1", "sid": "100", "tver": 0}
    token = create_access_token(claims, expires_delta=timedelta(minutes=5))

    # Decode and verify claims
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "1"
    assert decoded["sid"] == "100"
    assert decoded["tver"] == 0

    # Test expired token
    expired_token = create_access_token(claims, expires_delta=timedelta(seconds=-10))
    decoded_expired = decode_access_token(expired_token)
    assert decoded_expired is None


# ── Registration REST API Tests ────────────────────────────────────────── #

def test_register_flow():
    """Test user registration API endpoint."""
    payload = {
        "email": "newuser@convincesense.com",
        "password": "strongpassword123",
        "name": "New User",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["success"] is True

    # Test duplicate registration fails
    dup_response = client.post("/api/v1/auth/register", json=payload)
    assert dup_response.status_code == status.HTTP_400_BAD_REQUEST


# ── Login REST API Tests ───────────────────────────────────────────────── #

def test_login_flow():
    """Test login credentials verification and cookie/response payload."""
    # First register user
    register_payload = {
        "email": "userlogin@convincesense.com",
        "password": "loginpassword123",
        "name": "Login User",
    }
    client.post("/api/v1/auth/register", json=register_payload)

    # Correct credentials
    login_payload = {
        "email": "userlogin@convincesense.com",
        "password": "loginpassword123",
    }
    response = client.post("/api/v1/auth/login", json=login_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Verify response contains the secure cookies
    assert "access_token" in response.cookies
    assert "refresh_token" in response.cookies

    # Incorrect credentials
    invalid_payload = {
        "email": "userlogin@convincesense.com",
        "password": "wrongpassword",
    }
    invalid_response = client.post("/api/v1/auth/login", json=invalid_payload)
    assert invalid_response.status_code == status.HTTP_401_UNAUTHORIZED


# ── RBAC Authorization Dependency Tests ────────────────────────────────── #

def test_rbac_dependency():
    """Test accessing protected route with proper, missing, and invalid permissions."""
    db = TestingSessionLocal()

    # Create admin user
    admin_pwd_hash = hash_password("adminpass")
    admin_user = models.User(
        email="admin@convincesense.com",
        password_hash=admin_pwd_hash,
        is_active=True,
    )
    db.add(admin_user)
    db.flush()

    # Assign Admin role
    admin_role = db.query(models.Role).filter_by(name="admin").first()
    db.add(models.UserRole(user_id=admin_user.id, role_id=admin_role.id))

    # Create basic user
    user_pwd_hash = hash_password("userpass")
    basic_user = models.User(
        email="user@convincesense.com",
        password_hash=user_pwd_hash,
        is_active=True,
    )
    db.add(basic_user)
    db.flush()

    # Assign no roles (empty permissions)
    # create a session record for admin
    admin_session = models.Session(
        user_id=admin_user.id,
        refresh_token_hash="hash1",
        expires_at=models.datetime.utcnow() + timedelta(days=1),
    )
    db.add(admin_session)

    # create a session record for basic user
    user_session = models.Session(
        user_id=basic_user.id,
        refresh_token_hash="hash2",
        expires_at=models.datetime.utcnow() + timedelta(days=1),
    )
    db.add(user_session)

    db.commit()

    # Generate admin token
    admin_token = create_access_token(
        {"sub": str(admin_user.id), "sid": str(admin_session.id), "tver": 0}
    )
    # Generate basic user token
    user_token = create_access_token(
        {"sub": str(basic_user.id), "sid": str(user_session.id), "tver": 0}
    )

    db.close()

    # 1. Admin accesses /dummy-protected-route (has calls:write permission)
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.get("/dummy-protected-route", headers=headers)
    assert response.status_code == status.HTTP_200_OK

    # 2. Basic User accesses /dummy-protected-route (does not have permission)
    user_headers = {"Authorization": f"Bearer {user_token}"}
    response_denied = client.get("/dummy-protected-route", headers=user_headers)
    assert response_denied.status_code == status.HTTP_403_FORBIDDEN

    # 3. Access without token (cookies are cleared, so this should fail)
    response_unauth = client.get("/dummy-protected-route")
    assert response_unauth.status_code == status.HTTP_401_UNAUTHORIZED


# ── Session & Rotation Tests ───────────────────────────────────────────── #

def test_refresh_token_rotation():
    """Test refresh token rotation flow (Scenario A)."""
    # 1. Register & Login
    payload = {"email": "rotator@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload["email"], "password": payload["password"], "name": "Rotator"})
    login_resp = client.post("/api/v1/auth/login", json=payload)
    assert login_resp.status_code == status.HTTP_200_OK

    old_access_cookie = client.cookies.get("access_token")
    old_refresh_cookie = client.cookies.get("refresh_token")
    assert old_refresh_cookie is not None

    # 2. Call /refresh
    refresh_resp = client.post("/api/v1/auth/refresh")
    assert refresh_resp.status_code == status.HTTP_200_OK
    assert "access_token" in refresh_resp.json()

    new_access_cookie = client.cookies.get("access_token")
    new_refresh_cookie = client.cookies.get("refresh_token")
    assert new_refresh_cookie is not None
    assert new_refresh_cookie != old_refresh_cookie

    # 3. Check DB state
    db = TestingSessionLocal()
    # Old token hash should be rotated
    old_hash = hashlib.sha256(old_refresh_cookie.encode()).hexdigest()
    old_token_record = db.query(models.RefreshToken).filter_by(token_hash=old_hash).first()
    assert old_token_record is not None
    assert old_token_record.rotated is True

    # New token hash should exist and not rotated
    new_hash = hashlib.sha256(new_refresh_cookie.encode()).hexdigest()
    new_token_record = db.query(models.RefreshToken).filter_by(token_hash=new_hash).first()
    assert new_token_record is not None
    assert new_token_record.rotated is False
    db.close()


import hashlib

def test_refresh_token_reuse_breach_detection():
    """Test token theft/replay breach detection logic (Scenario B)."""
    # 1. Register & Login
    payload = {"email": "breached@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload["email"], "password": payload["password"], "name": "Breached"})
    login_resp = client.post("/api/v1/auth/login", json=payload)
    assert login_resp.status_code == status.HTTP_200_OK

    stolen_refresh_token = client.cookies.get("refresh_token")
    assert stolen_refresh_token is not None

    # 2. First rotation (valid use)
    first_refresh = client.post("/api/v1/auth/refresh")
    assert first_refresh.status_code == status.HTTP_200_OK
    valid_refresh_token = client.cookies.get("refresh_token")
    assert valid_refresh_token != stolen_refresh_token

    # 3. Replay attack: client uses the STOLEN (already rotated) token
    # Manually overwrite the refresh_token cookie with the stolen one
    client.cookies.set("refresh_token", stolen_refresh_token)

    replay_resp = client.post("/api/v1/auth/refresh")
    assert replay_resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert "security breach detected" in replay_resp.json()["detail"].lower()

    # 4. Verify DB state - all user sessions revoked
    db = TestingSessionLocal()
    user = db.query(models.User).filter_by(email=payload["email"]).first()
    assert user is not None
    
    sessions = db.query(models.Session).filter_by(user_id=user.id).all()
    assert len(sessions) > 0
    # Every single session must be revoked
    for s in sessions:
        assert s.revoked is True

    # Audit log of breach must be present
    audit = db.query(models.AuditLog).filter_by(user_id=user.id, event_type="REFRESH_TOKEN_REUSE_BREACH").first()
    assert audit is not None
    db.close()


def test_get_active_sessions():
    """Test retrieving active sessions and identifying the current active session."""
    payload = {"email": "sessionrep@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload["email"], "password": payload["password"], "name": "Session Rep"})
    login_resp = client.post("/api/v1/auth/login", json=payload)
    assert login_resp.status_code == status.HTTP_200_OK

    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Fetch sessions
    sessions_resp = client.get("/api/v1/auth/sessions", headers=headers)
    assert sessions_resp.status_code == status.HTTP_200_OK
    
    sessions_data = sessions_resp.json()
    assert len(sessions_data) == 1
    assert sessions_data[0]["is_current"] is True
    assert sessions_data[0]["device_name"] is not None


# ── Revocation & Lockout Tests ─────────────────────────────────────────── #

def test_logout():
    """Test standard logout revoking only the current session."""
    payload = {"email": "logouter@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload["email"], "password": payload["password"], "name": "Logouter"})
    login_resp = client.post("/api/v1/auth/login", json=payload)
    assert login_resp.status_code == status.HTTP_200_OK

    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Verify cookies exist
    assert "access_token" in client.cookies
    assert "refresh_token" in client.cookies

    # Call logout
    logout_resp = client.post("/api/v1/auth/logout", headers=headers)
    assert logout_resp.status_code == status.HTTP_200_OK
    assert logout_resp.json()["success"] is True

    # Verify cookies are cleared
    # Note: deletion cookies in FastAPI are returned in header, client will update its cookies
    assert "access_token" not in client.cookies
    assert "refresh_token" not in client.cookies

    # Verify session is revoked in DB
    db = TestingSessionLocal()
    user = db.query(models.User).filter_by(email=payload["email"]).first()
    session = db.query(models.Session).filter_by(user_id=user.id).first()
    assert session.revoked is True
    db.close()


def test_logout_all_devices():
    """Test logout-all-devices revoking all user sessions and incrementing token version."""
    payload = {"email": "logoutall@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload["email"], "password": payload["password"], "name": "Logout All"})
    
    # Login device 1
    login_resp1 = client.post("/api/v1/auth/login", json=payload)
    assert login_resp1.status_code == status.HTTP_200_OK
    token1 = login_resp1.json()["access_token"]

    # Clear cookies to simulate second device login
    client.cookies.clear()
    
    # Login device 2
    login_resp2 = client.post("/api/v1/auth/login", json=payload)
    assert login_resp2.status_code == status.HTTP_200_OK
    token2 = login_resp2.json()["access_token"]

    # Verify 2 active sessions exist
    db = TestingSessionLocal()
    user = db.query(models.User).filter_by(email=payload["email"]).first()
    active_sessions_count = db.query(models.Session).filter_by(user_id=user.id, revoked=False).count()
    assert active_sessions_count == 2
    assert user.token_version == 0
    db.close()

    # Call logout-all-devices with token2
    headers = {"Authorization": f"Bearer {token2}"}
    logout_resp = client.post("/api/v1/auth/logout-all-devices", headers=headers)
    assert logout_resp.status_code == status.HTTP_200_OK

    # Check DB state
    db = TestingSessionLocal()
    user = db.query(models.User).filter_by(email=payload["email"]).first()
    # token_version must be incremented to globally invalidate old JWTs
    assert user.token_version == 1

    # All sessions must be revoked
    all_sessions_revoked = db.query(models.Session).filter_by(user_id=user.id, revoked=False).count()
    assert all_sessions_revoked == 0
    db.close()

    # Verify that token1 is now rejected on a protected route because tver=0 < user.token_version=1
    headers1 = {"Authorization": f"Bearer {token1}"}
    resp1 = client.get("/dummy-protected-route", headers=headers1)
    assert resp1.status_code == status.HTTP_401_UNAUTHORIZED
    assert "session has expired" in resp1.json()["detail"].lower()


from datetime import datetime, timedelta

def test_account_lockout_sliding_window():
    """Test account lockout after max login attempts and automatic reset after lockout period."""
    email = "lockout_target@convincesense.com"
    pwd = "password123"
    client.post("/api/v1/auth/register", json={"email": email, "password": pwd, "name": "Lockout User"})

    # Perform 5 failed login attempts (max limit is 5)
    login_payload = {"email": email, "password": "wrongpassword"}
    for _ in range(5):
        resp = client.post("/api/v1/auth/login", json=login_payload)
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    # The 6th attempt should trigger 403 Account Locked
    locked_resp = client.post("/api/v1/auth/login", json=login_payload)
    assert locked_resp.status_code == status.HTTP_403_FORBIDDEN
    assert "temporarily locked" in locked_resp.json()["detail"].lower()

    # Verify another user can log in fine (lockout is email-specific)
    payload_other = {"email": "other@convincesense.com", "password": "password123"}
    client.post("/api/v1/auth/register", json={"email": payload_other["email"], "password": payload_other["password"], "name": "Other"})
    other_resp = client.post("/api/v1/auth/login", json=payload_other)
    assert other_resp.status_code == status.HTTP_200_OK

    # Simulate lockout duration passing (15 mins) by backdating the failed attempts in the database
    db = TestingSessionLocal()
    user = db.query(models.User).filter_by(email=email).first()
    attempts = db.query(models.LoginAttempt).filter_by(email=email, success=False).all()
    assert len(attempts) >= 5

    # Backdate attempts by 20 minutes
    for a in attempts:
        a.created_at = datetime.utcnow() - timedelta(minutes=20)
    db.commit()
    db.close()

    # A new attempt should now succeed (lockout has expired in the sliding window)
    success_payload = {"email": email, "password": pwd}
    retry_resp = client.post("/api/v1/auth/login", json=success_payload)
    assert retry_resp.status_code == status.HTTP_200_OK


