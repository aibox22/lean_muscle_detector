"""Failures the API can show to the caller."""


class Refuse(Exception):
    """The photo can be read, and it still must not be scored."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail or reason
        super().__init__(self.detail)


class GatewayError(Exception):
    """The gateway replied with something this pipeline cannot use."""


class MissingKey(Exception):
    """VLMRUN_API_KEY is not in the environment or the project .env."""
