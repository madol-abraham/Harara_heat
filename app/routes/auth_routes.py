# =============================================================================
# Authentication Routes for Harara Dashboard
# =============================================================================

from fastapi import APIRouter, HTTPException, Depends, status
from datetime import timedelta
from auth import (
    LoginRequest, TokenResponse, UserInfo, 
    authenticate_user, create_access_token, get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

@router.post("/login", response_model=TokenResponse, tags=["Authentication"])
async def login(login_data: LoginRequest):
    """Login endpoint - returns JWT token"""
    if not authenticate_user(login_data.username, login_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": login_data.username, "role": "admin"},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )

@router.get("/me", response_model=UserInfo, tags=["Authentication"])
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.post("/verify", tags=["Authentication"])
async def verify_token(current_user: UserInfo = Depends(get_current_user)):
    """Verify if token is valid"""
    return {"valid": True, "user": current_user}