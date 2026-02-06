from datetime import datetime
import os

def generate_run_id():
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")

def create_run_dirs(run_id):
    base_dir = os.path.join("artifacts", "runs", run_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def resolve_run_id(passed_run_id: str | None):
    return passed_run_id if passed_run_id else generate_run_id()
