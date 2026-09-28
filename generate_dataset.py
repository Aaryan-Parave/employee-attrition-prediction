"""
generate_dataset.py
-------------------
Generates a realistic synthetic version of the IBM HR Analytics Employee Attrition dataset.
The schema mirrors the original Kaggle dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv).
Run this ONCE to create data/HR_Employee_Attrition.csv before running the main analysis.
"""

import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

N = 1470  # Match the original IBM dataset size exactly

# ── Helper pools ─────────────────────────────────────────────────────────────
departments   = ["Sales", "Research & Development", "Human Resources"]
job_roles     = {
    "Sales":                    ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development":   ["Research Scientist", "Laboratory Technician",
                                  "Manufacturing Director", "Research Director", "Healthcare Representative"],
    "Human Resources":          ["Human Resources", "Manager"],
}
education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                     "Human Resources", "Other"]
marital_statuses = ["Single", "Married", "Divorced"]
business_travels = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]

# ── Base features (mostly continuous) ────────────────────────────────────────
age                 = np.random.randint(18, 61, N)
department_col      = np.random.choice(departments, N, p=[0.40, 0.46, 0.14])

# Job role depends on department
job_role_col = np.array([
    np.random.choice(job_roles[dept]) for dept in department_col
])

monthly_income      = np.random.randint(1009, 20000, N)
distance_from_home  = np.random.randint(1, 30, N)
education           = np.random.randint(1, 6, N)          # 1-Below College … 5-Doctor
education_field     = np.random.choice(education_fields, N)
environment_sat     = np.random.randint(1, 5, N)          # 1-Low … 4-Very High
gender              = np.random.choice(["Male", "Female"], N, p=[0.60, 0.40])
job_involvement     = np.random.randint(1, 5, N)
job_level           = np.random.randint(1, 6, N)
job_satisfaction    = np.random.randint(1, 5, N)          # KEY FEATURE
marital_status      = np.random.choice(marital_statuses, N, p=[0.32, 0.46, 0.22])
monthly_rate        = np.random.randint(2094, 27000, N)
num_companies_worked= np.random.randint(0, 10, N)
overtime            = np.random.choice(["Yes", "No"], N, p=[0.29, 0.71])  # KEY FEATURE
percent_salary_hike = np.random.randint(11, 26, N)
performance_rating  = np.random.choice([3, 4], N, p=[0.85, 0.15])
relationship_sat    = np.random.randint(1, 5, N)
stock_option_level  = np.random.randint(0, 4, N)
total_working_years = np.clip(np.random.randint(0, 41, N), 0, 40)
training_times      = np.random.randint(0, 7, N)
work_life_balance   = np.random.randint(1, 5, N)          # KEY FEATURE
years_at_company    = np.clip(np.random.randint(0, 41, N), 0, total_working_years)
years_in_role       = np.clip(np.random.randint(0, 19, N), 0, years_at_company)
years_since_promo   = np.clip(np.random.randint(0, 16, N), 0, years_at_company)
years_with_mgr      = np.clip(np.random.randint(0, 18, N), 0, years_at_company)
business_travel     = np.random.choice(business_travels, N, p=[0.19, 0.71, 0.10])

# ── Constant / ID columns (kept to mirror original dataset; dropped in analysis) ──
employee_number   = np.arange(1, N + 1)
employee_count    = np.ones(N, dtype=int)       # always 1 → zero variance
standard_hours    = np.full(N, 80, dtype=int)   # always 80 → zero variance
over_18           = np.full(N, "Y")             # always 'Y' → zero variance

# ── Target: Attrition (class-imbalanced, ~16% Yes like the real dataset) ─────
# We engineer attrition probability based on known risk factors so the model
# can actually learn meaningful patterns from the features.
attrition_prob = np.full(N, 0.10)  # base probability

attrition_prob += 0.18 * (overtime == "Yes")
attrition_prob += 0.10 * (job_satisfaction == 1)
attrition_prob += 0.06 * (job_satisfaction == 2)
attrition_prob -= 0.05 * (job_satisfaction == 4)
attrition_prob += 0.08 * (work_life_balance == 1)
attrition_prob += 0.06 * (distance_from_home > 20)
attrition_prob += 0.05 * (business_travel == "Travel_Frequently")
attrition_prob += 0.04 * (years_at_company < 2)
attrition_prob += 0.03 * (num_companies_worked > 6)
attrition_prob -= 0.05 * (stock_option_level > 1)
attrition_prob -= 0.04 * (job_level > 3)
attrition_prob += 0.05 * (age < 26)
attrition_prob -= 0.03 * (age > 45)
attrition_prob += 0.04 * (marital_status == "Single")
attrition_prob += 0.04 * (environment_sat == 1)
attrition_prob += 0.03 * (job_involvement == 1)
attrition_prob += 0.03 * (monthly_income < 3000)
attrition_prob -= 0.04 * (monthly_income > 15000)

# Clip probabilities to [0, 1]
attrition_prob = np.clip(attrition_prob, 0.01, 0.95)

# Sample binary attrition
attrition_raw = np.random.binomial(1, attrition_prob)
attrition_col = np.where(attrition_raw == 1, "Yes", "No")

print(f"Attrition distribution:\n  Yes: {(attrition_col == 'Yes').sum()} "
      f"({(attrition_col == 'Yes').mean()*100:.1f}%)\n"
      f"  No : {(attrition_col == 'No').sum()} "
      f"({(attrition_col == 'No').mean()*100:.1f}%)")

# ── Assemble DataFrame (column order mirrors original IBM dataset) ─────────────
df = pd.DataFrame({
    "Age":                     age,
    "Attrition":               attrition_col,
    "BusinessTravel":          business_travel,
    "DailyRate":               np.random.randint(102, 1500, N),
    "Department":              department_col,
    "DistanceFromHome":        distance_from_home,
    "Education":               education,
    "EducationField":          education_field,
    "EmployeeCount":           employee_count,
    "EmployeeNumber":          employee_number,
    "EnvironmentSatisfaction": environment_sat,
    "Gender":                  gender,
    "HourlyRate":              np.random.randint(30, 101, N),
    "JobInvolvement":          job_involvement,
    "JobLevel":                job_level,
    "JobRole":                 job_role_col,
    "JobSatisfaction":         job_satisfaction,
    "MaritalStatus":           marital_status,
    "MonthlyIncome":           monthly_income,
    "MonthlyRate":             monthly_rate,
    "NumCompaniesWorked":      num_companies_worked,
    "Over18":                  over_18,
    "OverTime":                overtime,
    "PercentSalaryHike":       percent_salary_hike,
    "PerformanceRating":       performance_rating,
    "RelationshipSatisfaction":relationship_sat,
    "StandardHours":           standard_hours,
    "StockOptionLevel":        stock_option_level,
    "TotalWorkingYears":       total_working_years,
    "TrainingTimesLastYear":   training_times,
    "WorkLifeBalance":         work_life_balance,
    "YearsAtCompany":          years_at_company,
    "YearsInCurrentRole":      years_in_role,
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrManager":    years_with_mgr,
})

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "HR_Employee_Attrition.csv")
df.to_csv(out_path, index=False)
print(f"\nDataset saved -> {out_path}")
print(f"Shape: {df.shape}")
 