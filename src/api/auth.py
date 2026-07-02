# src/api/auth.py
"""Authentication and registration REST endpoints."""

from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.auth.config import (
    ACCESS_TOKEN_COOKIE_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    COOKIE_DOMAIN,
    COOKIE_PATH,
    COOKIE_SECURE,
    COOKIE_SAMESITE,
    REFRESH_TOKEN_COOKIE_NAME,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from src.auth.security import create_access_token, hash_password, verify_password
from src.database import crud, models, schemas
from src.database.connection import get_db
from src.auth.dependencies import get_current_user

router = APIRouter()


def _hash_token(token: str) -> str:
    """Helper to generate a SHA-256 hash of a token for secure database storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def check_login_lockout(db: Session, email: str) -> None:
    """Check if the email has exceeded maximum login attempts and is temporarily locked out."""
    from src.auth.config import LOCKOUT_DURATION_MINUTES, MAX_LOGIN_ATTEMPTS

    # Get time of last successful login
    last_success_time = db.query(func.max(models.LoginAttempt.created_at)).filter(
        models.LoginAttempt.email == email,
        models.LoginAttempt.success == True
    ).scalar()

    # Query failed attempts since last success and within lockout window
    lockout_window_start = datetime.utcnow() - timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    if last_success_time:
        filter_start = max(last_success_time, lockout_window_start)
    else:
        filter_start = lockout_window_start

    failed_attempts = (
        db.query(models.LoginAttempt)
        .filter(
            models.LoginAttempt.email == email,
            models.LoginAttempt.success == False,
            models.LoginAttempt.created_at >= filter_start,
        )
        .order_by(models.LoginAttempt.created_at.desc())
        .all()
    )

    if len(failed_attempts) >= MAX_LOGIN_ATTEMPTS:
        last_failed = failed_attempts[0]
        unlock_time = last_failed.created_at + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        remaining = (unlock_time - datetime.utcnow()).total_seconds()
        if remaining > 0:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is temporarily locked due to too many failed login attempts. Please try again in {time_str}.",
            )


@router.post("/register", response_model=schemas.MessageResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: schemas.RegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new user account."""
    existing_user = db.query(models.User).filter_by(email=payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    # Create new user
    hashed_pwd = hash_password(payload.password)
    new_user = models.User(
        email=payload.email,
        name=payload.name,
        password_hash=hashed_pwd,
        is_active=True,
        is_verified=False,
    )
    db.add(new_user)
    db.flush()  # Populates user ID

    # Assign default 'user' role
    user_role = db.query(models.Role).filter_by(name="user").first()
    if user_role:
        db.add(models.UserRole(user_id=new_user.id, role_id=user_role.id))

    # Audit log
    crud.create_audit_log(
        db,
        schemas.AuditLogCreate(
            user_id=new_user.id,
            event_type="REGISTER_USER",
            details={"email": new_user.email, "name": new_user.name},
        ),
    )

    db.commit()
    return {"success": True, "message": "User registered successfully."}


@router.post("/login", response_model=schemas.TokenResponse)
def login(
    response: Response,
    request: Request,
    payload: schemas.LoginRequest,
    db: Session = Depends(get_db),
):
    """Authenticate credentials and establish a secure session."""
    # Check for account lockout before credential check
    check_login_lockout(db, payload.email)

    user = db.query(models.User).filter_by(email=payload.email).first()
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Lockout check placeholder (will be fully implemented in Phase 4)
    # Check if credentials are correct
    if not user or not user.password_hash or not verify_password(user.password_hash, payload.password):
        # Record failed login attempt
        failed_attempt = models.LoginAttempt(
            user_id=user.id if user else None,
            email=payload.email,
            success=False,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(failed_attempt)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account has been deactivated.",
        )

    # 1. Create a stateful session
    raw_refresh_token = secrets.token_urlsafe(64)
    hashed_refresh_token = _hash_token(raw_refresh_token)

    session_expiry = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Simple device name parser for session telemetry
    device_name = "Unknown Device"
    browser = "Unknown Browser"
    operating_system = "Unknown OS"
    if user_agent:
        # Very lightweight detection (Next.js BFF will pass cleaner info)
        if "Windows" in user_agent:
            operating_system = "Windows"
        elif "Macintosh" in user_agent:
            operating_system = "macOS"
        elif "Linux" in user_agent:
            operating_system = "Linux"

        if "Chrome" in user_agent:
            browser = "Chrome"
        elif "Safari" in user_agent:
            browser = "Safari"
        elif "Firefox" in user_agent:
            browser = "Firefox"

        device_name = f"{operating_system} - {browser}"

    db_session = models.Session(
        user_id=user.id,
        refresh_token_hash=hashed_refresh_token,
        device_name=device_name,
        browser=browser,
        operating_system=operating_system,
        ip_address=ip_address,
        expires_at=session_expiry,
    )
    db.add(db_session)
    db.flush()  # Populate db_session.id

    # Save to refresh_tokens table
    db_refresh = models.RefreshToken(
        session_id=db_session.id,
        user_id=user.id,
        token_hash=hashed_refresh_token,
        expires_at=session_expiry,
    )
    db.add(db_refresh)

    # Record login attempt
    success_attempt = models.LoginAttempt(
        user_id=user.id,
        email=user.email,
        success=True,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(success_attempt)

    # Audit log
    crud.create_audit_log(
        db,
        schemas.AuditLogCreate(
            user_id=user.id,
            event_type="LOGIN_SUCCESS",
            details={"session_id": db_session.id, "ip": ip_address},
        ),
    )

    db.commit()

    # 2. Generate lightweight JWT access token
    access_token_claims = {
        "sub": str(user.id),
        "sid": str(db_session.id),
        "tver": user.token_version,
    }
    access_token = create_access_token(
        data=access_token_claims,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    # 3. Set secure HTTP-only cookies
    cookie_settings = {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": COOKIE_PATH,
    }
    if COOKIE_DOMAIN:
        cookie_settings["domain"] = COOKIE_DOMAIN

    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **cookie_settings,
    )

    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=raw_refresh_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **cookie_settings,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=schemas.RefreshResponse)
def refresh(
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    """Rotate the refresh token and issue a new access token."""
    raw_refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE_NAME)
    if not raw_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token cookie.",
        )

    hashed_token = _hash_token(raw_refresh_token)
    db_refresh = db.query(models.RefreshToken).filter_by(token_hash=hashed_token).first()

    # Theft / Replay detection:
    if db_refresh and db_refresh.rotated:
        # Crucial security breach detection: token has already been rotated!
        # Immediately revoke all active sessions for this user.
        db.query(models.Session).filter(
            models.Session.user_id == db_refresh.user_id
        ).update({"revoked": True})
        
        crud.create_audit_log(
            db,
            schemas.AuditLogCreate(
                user_id=db_refresh.user_id,
                event_type="REFRESH_TOKEN_REUSE_BREACH",
                details={
                    "session_id": db_refresh.session_id,
                    "ip": request.client.host if request.client else None,
                },
            ),
        )
        db.commit()

        # Clear cookies
        response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)
        response.delete_cookie(REFRESH_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Security breach detected. All active sessions have been revoked. Please log in again.",
        )

    # Validate token
    if not db_refresh or db_refresh.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    # Check parent session status
    db_session = db.query(models.Session).filter_by(id=db_refresh.session_id).first()
    if not db_session or db_session.revoked or db_session.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked or expired.",
        )

    # Validate active user
    user = db.query(models.User).filter_by(id=db_refresh.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or not found.",
        )

    # Perform Token Rotation:
    # 1. Mark old token as rotated
    db_refresh.rotated = True

    # 2. Generate new refresh token
    new_raw_refresh_token = secrets.token_urlsafe(64)
    new_hashed_refresh_token = _hash_token(new_raw_refresh_token)
    new_expiry = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # 3. Save new RefreshToken in DB
    new_db_refresh = models.RefreshToken(
        session_id=db_session.id,
        user_id=user.id,
        token_hash=new_hashed_refresh_token,
        expires_at=new_expiry,
    )
    db.add(new_db_refresh)

    # 4. Update the Session token and last_active
    db_session.refresh_token_hash = new_hashed_refresh_token
    db_session.last_active = datetime.utcnow()

    # 5. Log the rotation audit
    crud.create_audit_log(
        db,
        schemas.AuditLogCreate(
            user_id=user.id,
            event_type="REFRESH_TOKEN_ROTATION",
            details={"session_id": db_session.id},
        ),
    )

    db.commit()

    # 6. Generate new access token
    access_token_claims = {
        "sub": str(user.id),
        "sid": str(db_session.id),
        "tver": user.token_version,
    }
    new_access_token = create_access_token(
        data=access_token_claims,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    # 7. Set cookies
    cookie_settings = {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": COOKIE_PATH,
    }
    if COOKIE_DOMAIN:
        cookie_settings["domain"] = COOKIE_DOMAIN

    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=new_access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **cookie_settings,
    )

    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=new_raw_refresh_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **cookie_settings,
    )

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
    }


@router.get("/sessions", response_model=list[schemas.SessionInfo])
def get_user_sessions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Retrieve all active sessions for the current authenticated user."""
    sessions = (
        db.query(models.Session)
        .filter_by(user_id=current_user.id, revoked=False)
        .filter(models.Session.expires_at > datetime.utcnow())
        .order_by(models.Session.last_active.desc())
        .all()
    )

    session_infos = []
    for s in sessions:
        is_current = current_user.current_session_id == s.id
        session_infos.append(
            schemas.SessionInfo(
                id=s.id,
                device_name=s.device_name,
                browser=s.browser,
                operating_system=s.operating_system,
                ip_address=s.ip_address,
                approximate_location=s.approximate_location,
                last_active=s.last_active,
                created_at=s.created_at,
                is_current=is_current,
            )
        )
    return session_infos


@router.post("/logout", response_model=schemas.MessageResponse)
def logout(
    response: Response,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Revoke the current active session and clear cookies."""
    if current_user.current_session_id:
        db_session = db.query(models.Session).filter_by(id=current_user.current_session_id).first()
        if db_session:
            db_session.revoked = True
            
            crud.create_audit_log(
                db,
                schemas.AuditLogCreate(
                    user_id=current_user.id,
                    event_type="LOGOUT_SUCCESS",
                    details={"session_id": db_session.id},
                ),
            )
            db.commit()

    # Clear access and refresh token cookies
    response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)
    response.delete_cookie(REFRESH_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)

    return {"success": True, "message": "Logged out successfully."}


@router.post("/logout-all-devices", response_model=schemas.MessageResponse)
def logout_all_devices(
    response: Response,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Revoke all active sessions and globally invalidate all issued access tokens for the user."""
    # 1. Invalidate all sessions in DB
    db.query(models.Session).filter_by(user_id=current_user.id).update({"revoked": True})

    # 2. Increment token version for immediate JWT revocation
    current_user.token_version += 1
    
    # 3. Audit log
    crud.create_audit_log(
        db,
        schemas.AuditLogCreate(
            user_id=current_user.id,
            event_type="LOGOUT_ALL_DEVICES_SUCCESS",
            details={"new_token_version": current_user.token_version},
        ),
    )
    db.commit()

    # Clear access and refresh token cookies
    response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)
    response.delete_cookie(REFRESH_TOKEN_COOKIE_NAME, path=COOKIE_PATH, domain=COOKIE_DOMAIN)

    return {"success": True, "message": "Successfully logged out of all devices."}
