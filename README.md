# Gym Members Exercise Analysis

A data analysis project exploring gym member workout patterns, physical attributes, and fitness metrics. This project demonstrates the complete data analysis workflow from exploratory analysis through statistical testing to actionable business recommendations.

## Project Status

**Complete** — All phases finished: EDA, visualization, feature engineering, statistical testing, and business recommendations.

## Project Overview

This completed analysis examines a dataset of 973 gym members, investigating relationships between workout characteristics, physical attributes, and performance metrics. The notebook now includes all major stages of the workflow and addresses questions such as:

- What factors most strongly predict caloric expenditure during workouts?
- How do performance metrics vary across experience levels and workout types?
- Are observed differences between demographic groups statistically significant?

### Key Finding

Session duration emerges as the dominant predictor of calories burned, with a correlation coefficient of 0.91, accounting for approximately 83% of the variance in caloric expenditure. Experience level shows systematic progression in energy output, while workout type demonstrates no statistically significant difference in calorie burn.

## Technical Approach

### Data Quality Assessment

The analysis begins with data profiling, including structural validation, missing value detection, and categorical integrity checks. I identified and documented that this dataset is synthetic, generated from published research averages. This recognition informed all subsequent analysis and conclusions.

### Exploratory Data Analysis

The EDA phase covers the analysis steps presented in the notebook:

**Data Profiling**: Structural validation, missing value checks, and initial descriptive statistics to confirm the dataset was clean and properly formatted.

**Recognizing the Data Source & Context**: Acknowledging that the dataset is synthetic and discussing the implications for interpretation and generalization.

**Data Preparation**: Converting key measurements from metric to imperial units for broader interpretability and ensuring the dataset is ready for analysis.

**Data Visualization**: Including univariate, categorical, and bivariate analyses through histograms, count plots, heatmaps, box plots, and scatter plots.

### Feature Engineering

Created derived features and unit conversions to support deeper analysis:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `Calorie_Efficiency` | Calories burned per hour | Normalize performance across session lengths |
| `BMI_Category` | WHO standard classifications | Enable health-risk segmentation |
| `Age_Group` | Life-stage groupings | Facilitate demographic analysis |
| `Intensity_Score` | Heart rate capacity utilization | Quantify workout effort |

### Statistical Methods

- Descriptive statistics and distribution analysis
- Correlation analysis (Pearson)
- Data standardization (StandardScaler)
- Independent samples t-test
- One-way ANOVA

## Repository Structure

```
├── gym_members_analysis.ipynb    # Main analysis notebook
├── gym_members_analysis.py       # Python script version (Jupytext sync)
├── data/
│   └── gym_members_exercise_tracking.csv
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

## Tools and Technologies

**Languages**: Python 3.13

**Data Manipulation**: pandas, NumPy

**Visualization**: Matplotlib, Seaborn

**Statistical Analysis**: SciPy, scikit-learn

**Development Environment**: Jupyter Notebook, VS Code

## Data Source

Dataset sourced from Kaggle: [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

The dataset is synthetic, generated using averages from publicly available fitness studies and industry reports. All conclusions acknowledge this limitation and note that findings would require validation against real-world data before practical application.

## What This Project Demonstrates

- Systematic approach to data exploration with documented methodology
- Recognition of data limitations and appropriate caveats on conclusions
- Application of Python data science tools including pandas, seaborn, matplotlib, and scikit-learn
- Clear documentation of findings with visualizations designed for both technical and general audiences
- Statistical testing and business-oriented recommendations grounded in the analysis
- Version control with Git and organized project structure

## About the Author

I am building skills in Data Analysis with a background in Business Administration and Marketing. This project represents my first step into the Data Science field and my approach to learning: methodical, well-documented, and focused on practical skills.

I welcome feedback from experienced data professionals.

## Connect

**LinkedIn**: [linkedin.com/in/axelcabato](https://linkedin.com/in/axelcabato)

**Email**: contact@axelcabato.com

---

*Last updated: July 2026*