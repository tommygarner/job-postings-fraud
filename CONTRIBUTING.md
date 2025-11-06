# Contributing Guidelines

## Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation
- `exp/description` - Experiments

### 2. Make Changes
- Write clear, commented code
- Follow PEP 8 style
- Add docstrings to functions
- Keep commits atomic

### 3. Commit & Push
```bash
git add .
git commit -m "Clear description"
git push origin feature/your-feature-name
```

### 4. Create Pull Request
- Describe what changed and why
- Request review from teammate
- Allow time for feedback

## Code Style

Use snake_case for files and functions:
```python
def calculate_completeness_score(posting):
    """Calculate data completeness (0-10 scale)."""
    important_fields = ['salary_range', 'department', ...]
    return sum(~posting[important_fields].isna())
```

## Notebook Conventions

1. **Naming**: `##_description.ipynb` (01_eda, 02_feature_engineering)
2. **Structure**: Clear headers, markdown explanations
3. **Cleaning**: Remove outputs before committing

## Questions?

Create a GitHub Issue!
