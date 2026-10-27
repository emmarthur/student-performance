# Book directions — Student performance prediction

**Catalog #1186** · Joshua Eckroth, *Python Artificial Intelligence Projects for Beginners*, **Chapter 1** — student performance / decision trees

- Packt repo: [Chapter01/Student performance.py](https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/tree/master/Chapter01)
- Local reference: [`reference/Student_performance.py`](./reference/Student_performance.py)
- Shared logic: [`student_performance_core.py`](./student_performance_core.py)
- Dataset: [`data/student-por.csv`](./data/student-por.csv) (UCI student performance, Portuguese)

## What the book asks you to build

Predict whether a student **passes** using a **decision tree**:

1. Load `student-por.csv` (semicolon-separated).
2. Label **pass** = 1 when `G1 + G2 + G3 >= 35`, else 0; drop grade columns from features.
3. **One-hot encode** categorical columns (`sex`, `school`, jobs, support flags, etc.).
4. Shuffle; train on the **first 500** rows, test on the remainder.
5. `DecisionTreeClassifier(criterion="entropy", max_depth=5)`; report holdout score and **5-fold CV**.

Optional: export Graphviz `.dot`, sweep `max_depth` 1–19.

## Web version (`web/` — FastAPI, port **3028**)

| Book step | Web equivalent |
|-----------|----------------|
| Prepare data + train tree | `student_performance_core.get_or_train_model` (cached) |
| Evaluate | `GET /api/status` (holdout + CV) |
| Classify a profile | `POST /api/predict` with raw field values |
| Explore rows | Sample profile chips in UI |

CLI: train and score via `python -c "from student_performance_core import get_or_train_model; print(get_or_train_model().holdout_accuracy)"`
