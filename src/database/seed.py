# src/database/seed.py
"""Database seed script for default roles, permissions, and admin user.

Idempotent — safe to run multiple times without duplicating data.

Usage:
    python -m src.database.seed
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.database.models import (
    Permission,
    Role,
    RolePermission,
    User,
    UserRole,
)

# ── Default Data ─────────────────────────────────────────────────────── #

DEFAULT_ROLES = [
    {"name": "admin", "description": "Full system administrator"},
    {"name": "user", "description": "Standard authenticated user"},
]

DEFAULT_PERMISSIONS = [
    {"code": "users:read", "description": "View user profiles"},
    {"code": "users:write", "description": "Create and edit users"},
    {"code": "users:delete", "description": "Delete user accounts"},
    {"code": "calls:read", "description": "View call recordings and data"},
    {"code": "calls:write", "description": "Create and edit calls"},
    {"code": "calls:delete", "description": "Delete calls"},
    {"code": "analytics:read", "description": "Access analytics dashboard"},
    {"code": "settings:read", "description": "View system settings"},
    {"code": "settings:write", "description": "Modify system settings"},
    {"code": "audit:read", "description": "View audit logs"},
]

ROLE_PERMISSION_MAP = {
    "admin": [
        "users:read", "users:write", "users:delete",
        "calls:read", "calls:write", "calls:delete",
        "analytics:read",
        "settings:read", "settings:write",
        "audit:read",
    ],
    "user": [
        "users:read",
        "calls:read", "calls:write", "calls:delete",
        "analytics:read",
    ],
}


def seed_roles_and_permissions(db: Optional[Session] = None) -> dict:
    """Seed default roles and permissions into the database.

    Idempotent: skips any role/permission that already exists.

    Args:
        db: Optional SQLAlchemy session. Creates one if not provided.

    Returns:
        Summary dict with counts of created vs. existing items.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    summary = {
        "roles_created": 0,
        "roles_existed": 0,
        "permissions_created": 0,
        "permissions_existed": 0,
        "mappings_created": 0,
        "mappings_existed": 0,
    }

    try:
        # Seed permissions
        for perm_data in DEFAULT_PERMISSIONS:
            existing = db.query(Permission).filter_by(code=perm_data["code"]).first()
            if existing:
                summary["permissions_existed"] += 1
                print(f"  [skip] Permission '{perm_data['code']}' already exists")
            else:
                db.add(Permission(**perm_data))
                summary["permissions_created"] += 1
                print(f"  [+] Created permission '{perm_data['code']}'")

        # Seed roles
        for role_data in DEFAULT_ROLES:
            existing = db.query(Role).filter_by(name=role_data["name"]).first()
            if existing:
                summary["roles_existed"] += 1
                print(f"  [skip] Role '{role_data['name']}' already exists")
            else:
                db.add(Role(**role_data))
                summary["roles_created"] += 1
                print(f"  [+] Created role '{role_data['name']}'")

        db.flush()  # Ensure IDs are populated before mapping

        # Seed role-permission mappings
        for role_name, perm_codes in ROLE_PERMISSION_MAP.items():
            role = db.query(Role).filter_by(name=role_name).first()
            if not role:
                print(f"  [!] Role '{role_name}' not found, skipping mappings")
                continue

            for perm_code in perm_codes:
                permission = db.query(Permission).filter_by(code=perm_code).first()
                if not permission:
                    print(f"  [!] Permission '{perm_code}' not found, skipping")
                    continue

                existing_mapping = db.query(RolePermission).filter_by(
                    role_id=role.id, permission_id=permission.id
                ).first()
                if existing_mapping:
                    summary["mappings_existed"] += 1
                else:
                    db.add(RolePermission(role_id=role.id, permission_id=permission.id))
                    summary["mappings_created"] += 1
                    print(f"  [+] Mapped '{role_name}' → '{perm_code}'")

        db.commit()
        print("\n✅ Seed completed successfully.")
        print(f"   Roles:       {summary['roles_created']} created, {summary['roles_existed']} already existed")
        print(f"   Permissions: {summary['permissions_created']} created, {summary['permissions_existed']} already existed")
        print(f"   Mappings:    {summary['mappings_created']} created, {summary['mappings_existed']} already existed")

    except Exception as e:
        db.rollback()
        print(f"\n❌ Seed failed: {e}")
        raise
    finally:
        if close_session:
            db.close()

    return summary


def seed_default_admin_user(
    email: str,
    password_hash: str,
    name: str = "Admin",
    db: Optional[Session] = None,
) -> dict:
    """Create a default admin user with the admin role.

    Idempotent: skips if a user with the given email already exists.

    Args:
        email: Admin email address.
        password_hash: Pre-hashed password (hashing is NOT done here).
        name: Display name for the admin user.
        db: Optional SQLAlchemy session.

    Returns:
        Summary dict.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            print(f"  [skip] Admin user '{email}' already exists")
            return {"created": False, "user_id": existing.id}

        admin_user = User(
            email=email,
            name=name,
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
        )
        db.add(admin_user)
        db.flush()

        # Assign admin role
        admin_role = db.query(Role).filter_by(name="admin").first()
        if admin_role:
            db.add(UserRole(user_id=admin_user.id, role_id=admin_role.id))
            print(f"  [+] Assigned 'admin' role to '{email}'")
        else:
            print("  [!] 'admin' role not found. Run seed_roles_and_permissions() first.")

        db.commit()
        print(f"  [+] Created admin user '{email}' (id={admin_user.id})")
        return {"created": True, "user_id": admin_user.id}

    except Exception as e:
        db.rollback()
        print(f"  ❌ Failed to create admin user: {e}")
        raise
    finally:
        if close_session:
            db.close()


if __name__ == "__main__":
    print("🌱 Seeding ConvinceSense database...\n")
    print("── Roles & Permissions ──")
    seed_roles_and_permissions()
