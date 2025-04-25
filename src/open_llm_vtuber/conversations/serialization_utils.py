from typing import Any, Union, Dict # Added Dict for clarity

# Fix serialization for display_text
def serialize_display_text(display_text: Any) -> Union[dict, str]:
    """Safely serialize display_text for JSON transmission"""
    try:
        if hasattr(display_text, "__dict__"):
            return vars(display_text)
        elif hasattr(display_text, "dict"):
            return display_text.dict()
        else:
            return str(display_text)
    except Exception:
        return str(display_text)