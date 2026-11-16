import os

# Tests never read or write the developer's local training library.
os.environ.setdefault("FORECAST_LIBRARY_ENABLED", "false")
