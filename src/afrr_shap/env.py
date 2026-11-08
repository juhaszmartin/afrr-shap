"""Loading of the project-local .env file that holds the data-source API keys."""

from dotenv import load_dotenv

from afrr_shap.paths import PROJECT_ROOT


def load_env():
    """Load .env into the process environment; variables already set take precedence."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
