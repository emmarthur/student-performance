# Student performance predictor (catalog #1186)

Decision-tree pass/fail model from Joshua Eckroth’s *Python Artificial Intelligence Projects for Beginners* (Chapter 1). Logic in `student_performance_core.py`.

## Run the web demo (port 3028)

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\student-performance\web"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 3028
```

Open http://localhost:3028

Book provenance: [BOOK_DIRECTIONS.md](./BOOK_DIRECTIONS.md).
