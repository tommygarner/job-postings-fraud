# Dataset Information

## Source
Kaggle: Real or Fake Fake Job Posting Prediction  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Statistics
- **Total Records**: 17,880
- **Fraudulent**: 866 (4.84%)
- **Legitimate**: 17,014 (95.16%)
- **Features**: 18 columns + 1 target

## Features

### Text Features
- `title`: Job title
- `description`: Full job description
- `requirements`: Skills/requirements
- `benefits`: Employee benefits
- `company_profile`: Company overview

### Categorical Features
- `location`: Job location
- `employment_type`: Full-time, Part-time, etc.
- `industry`: Business industry
- `function`: Job function
- `required_experience`: Experience level
- `required_education`: Education requirement

### Binary Features
- `telecommuting`: Allows remote work (0/1)
- `has_company_logo`: Company logo provided (0/1)
- `has_questions`: Posting includes questions (0/1)

### Target
- `fraudulent`: 1 if fraud, 0 if legitimate

## Top Missing Values
| Feature | Missing % |
|---------|-----------|
| salary_range | 83.96% |
| department | 64.58% |
| required_education | 45.33% |
| benefits | 40.34% |
| required_experience | 39.43% |
| function | 36.10% |
| industry | 27.42% |

## Key Insight
Missingness itself is a fraud signal! (83% missing salary, 65% missing department)
