from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User, Workspace
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    model_config = {"json_schema_extra": {"example": {"email": "alice@example.com", "password": "s3cret!"}}}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    workspace_id: int


class UserOut(BaseModel):
    id: int
    email: str
    workspace_id: int

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/register", response_model=UserOut, status_code=201)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_db),
) -> UserOut:
    # Check duplicate email
    existing = await session.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Each user gets their own isolated workspace
    workspace = Workspace(name=f"ws-{body.email}")
    session.add(workspace)
    await session.flush()  # get workspace.id

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        workspace_id=workspace.id,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    log.info("Registered user email=%s workspace_id=%d", body.email, workspace.id)
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    session: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await session.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(user_id=user.id, workspace_id=user.workspace_id)
    log.info("Login user_id=%d workspace_id=%d", user.id, user.workspace_id)
    return TokenResponse(access_token=token, workspace_id=user.workspace_id)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
