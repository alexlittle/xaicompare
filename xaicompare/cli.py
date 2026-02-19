
import typer
from .runner import publish_run

app = typer.Typer()

@app.command()
def report(run_dir: str):
    import os
    os.system(f"streamlit run -q - " + \
              f"<<'EOF'\nfrom xaicompare.dashboard.app import main\nmain()\nEOF")

if __name__ == "__main__":
    app()
