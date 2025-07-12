class DataLoadingException(Exception):
    """Exception raised when data loading fails."""
    pass

class ClassificationException(Exception):
    """Exception raised when classification fails."""
    pass

class DataSavingException(Exception):
    """Exception raised when saving data fails."""
    pass


class OllamaAnalysisException(Exception):
    """Exception raised when Ollama analysis fails."""
    pass

class AnalysisException(Exception):
    """Exception raised for general analysis errors."""
    pass

class DataAnalysisException(Exception):
    """Exception raised when data analysis fails."""
    pass
