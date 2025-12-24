import requests
from typing import Optional, Dict, Any
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def score_cml_data(uploaded_file) -> Optional[Dict[str, Any]]:
    """Score CML data via API.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        API response as dict or None on error
    """
    try:
        # Robust MIME type handling
        mime_type = (
            uploaded_file.type if uploaded_file.type else "application/octet-stream"
        )

        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}

        # Log upload attempt
        print(
            f"Uploading {uploaded_file.name} ({len(uploaded_file.getvalue())} bytes, type={mime_type})"
        )

        response = requests.post(
            f"{API_BASE_URL}/score-cml-data", files=files, timeout=60
        )

        if response.status_code == 422:
            st.error(f"Validation Error: {response.text}")
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API. Make sure the API server is running on http://localhost:8000"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        try:
            # Try to print more detail if available
            st.code(response.text)
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None
