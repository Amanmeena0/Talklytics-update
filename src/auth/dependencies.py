# src/auth/dependencies.py
"""Authentication and authorization FastAPI dependencies."""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.auth.config import ACCESS_TOKEN_COOKIE_NAME
from src.auth.security import decode_access_token
from src.database.connection import get_db
from src.database.models import Permission, RolePermission, User, UserRole

# Define OAuth2 bearer scheme, making it optional so we can fall back to cookies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_token_from_request(
    request: Request,
    header_token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """Retrieve access token from Authorization header or HTTP-only cookies."""
    if header_token:
        return header_token

    # Check cookies as fallback
    cookie_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if cookie_token:
        return cookie_token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Missing token in header or cookie.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(get_token_from_request),
) -> User:
    """Validate access token and return the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str: Optional[str] = payload.get("sub")
    token_version: Optional[int] = payload.get("tver")

    if user_id_str is None or token_version is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise credentials_exception

    user = db.query(User).filter_by(id=user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    if token_version < user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Attach current session ID for active session check
    try:
        user.current_session_id = int(payload.get("sid"))
    except (TypeError, ValueError):
        user.current_session_id = None

    return user


def check_user_permission(db: Session, user_id: int, permission_code: str) -> bool:
    """Check if a user has a specific permission code via their assigned roles."""
    # Query user_roles -> roles -> role_permissions -> permissions
    has_perm = (
        db.query(Permission)
        .join(RolePermission, RolePermission.permission_id == Permission.id)
        .join(UserRole, UserRole.role_id == RolePermission.role_id)
        .filter(
            UserRole.user_id == user_id,
            Permission.code == permission_code,
        )
        .first()
        is not None
    )
    return has_perm


def require_permission(permission_code: str):
    """Factory dependency to enforce role-based access control (RBAC)."""

    def dependency(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> User:
        if not check_user_permission(db, current_user.id, permission_code):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Requires '{permission_code}'.",
            )
        return current_user

    return dependency
