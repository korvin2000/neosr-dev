"""Configuration errors (extended by the config loader). No torch import."""


class ConfigError(ValueError):
    """Invalid configuration: wrong/missing key, unknown type or removed feature (exit code 2)."""


class ConfigWarning(UserWarning):
    """Suspicious but accepted configuration, e.g. an unknown key."""
