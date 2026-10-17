from typing import List, Any, Tuple, Optional
from common.dataset import DataItem

def make_hashable(obj: Any) -> Any:
    """
    Standard hashable conversion (used for BIRD/Spider).
    """
    if hasattr(obj, "tolist") and callable(obj.tolist):
        obj = obj.tolist()
        
    if isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    
    return obj

def get_execution_result_hash(data_item: DataItem, result_rows: Optional[List[Any]]) -> Any:
    """
    Get the appropriate hashable representation of execution results based on the dataset.
    """
    if result_rows is None:
        return None
        
    # Standard BIRD/Spider logic: frozenset of rows
    return frozenset(make_hashable(result_rows))
