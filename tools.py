# tools.py
# Mock tool for lead capture

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.
    In production, this would POST to a CRM or database.
    """
    print("\n" + "="*50)
    print("✅ LEAD CAPTURED SUCCESSFULLY")
    print("="*50)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print("="*50 + "\n")

    return {
        "status": "success",
        "message": f"Lead captured for {name}",
        "data": {
            "name": name,
            "email": email,
            "platform": platform
        }
    }
