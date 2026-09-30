def a():
    try:
        pass
    except Exception:
        raise

def b():
    try:
        pass
    except Exception as exc:
        row = {"status": "error", "error": str(exc)}
    return row

def c():
    try:
        pass
    except Exception:
        return []
