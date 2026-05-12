"""Project-specific exceptions."""


class FWIError(Exception):
    """Base class for package-specific exceptions."""


class ConfigError(FWIError):
    """Raised when configuration loading or validation fails."""


class CredentialError(FWIError):
    """Raised when CDS credentials are missing or unusable."""


class DownloadError(FWIError):
    """Raised when a CDS request fails."""


class DatasetUnavailableError(DownloadError):
    """Raised when requested dates exceed dataset availability."""


class ProcessingError(FWIError):
    """Raised when a processing window cannot be completed."""


class CacheError(FWIError):
    """Raised when cache metadata cannot be read or written."""