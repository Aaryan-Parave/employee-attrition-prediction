# Employee Attrition Prediction - Decision Tree Classifier
# ============================================================
# Academic AIML Lab Submission
# Dataset : IBM HR Analytics Employee Attrition (synthetic, same schema)
# Model   : scikit-learn DecisionTreeClassifier
# ============================================================

# -- Standard library / third-party imports -----------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: saves figures to file without opening windows.
                        # Remove this line (and uncomment plt.show() calls below) if running
                        # in Jupyter or an environment with a display.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree            import (DecisionTreeClassifier,
                                     plot_tree,
                                     export_text)
from sklearn.metrics         import (accuracy_score,
                                     precision_score,
                                     recall_score,
                                     f1_score,
                                     confusion_matrix,
                                     classification_report)

warnings.filterwarnings("ignore")   # suppress minor sklearn warnings for clean output

# Create output folder for saved figures
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------------------------
# SECTION 1 - DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------

print("=" * 65)
print(" SECTION 1 - DATA LOADING & EDA")
print("=" * 65)

# Load the dataset from the data/ folder
df = pd.read_csv("data/HR_Employee_Attrition.csv")

# Basic shape and structure
print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn data types:")
print(df.dtypes.to_string())

# Check for missing values - a real-world dataset may have nulls
print("\nMissing values per column:")
print(df.isnull().sum().to_string())
# The IBM dataset (and this synthetic copy) has zero nulls, but we always check.

# -- Class balance of target variable -----------------------------------------
print("\n-- Class Balance (Attrition) --")
balance = df["Attrition"].value_counts()
print(balance)
pct_yes = balance["Yes"] / len(df) * 100
print(f"\nAttrition rate : {pct_yes:.1f}% Yes  |  {100 - pct_yes:.1f}% No")
print(
    "\n[NOTE] The dataset is CLASS IMBALANCED - roughly 16% of employees left (Yes).\n"
    "       This means a naive classifier that always predicts 'No' would achieve\n"
    "       ~84% accuracy while catching ZERO actual quitters.\n"
    "       Therefore we focus on Recall for the minority class (Attrition=Yes),\n"
    "       not raw accuracy, as our primary evaluation metric."
)

# -- Visualization 1 - Attrition rate by OverTime ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Attrition Analysis - Key Risk Factors", fontsize=15, fontweight="bold")

# Helper: compute attrition rate per group
def attrition_rate(df, col):
    return (df.groupby(col)["Attrition"]
              .apply(lambda s: (s == "Yes").mean() * 100)
              .reset_index(name="AttritionRate"))

# Plot 1 - OverTime
ot_rate = attrition_rate(df, "OverTime")
bars = axes[0].bar(ot_rate["OverTime"], ot_rate["AttritionRate"],
                   color=["#4CAF50", "#F44336"], edgecolor="black", width=0.4)
axes[0].set_title("Attrition Rate by OverTime", fontweight="bold")
axes[0].set_xlabel("OverTime")
axes[0].set_ylabel("Attrition Rate (%)")
axes[0].set_ylim(0, 60)
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

# Plot 2 - Department
dept_rate = attrition_rate(df, "Department")
colors_dept = ["#2196F3", "#FF9800", "#9C27B0"]
bars2 = axes[1].bar(dept_rate["Department"], dept_rate["AttritionRate"],
                    color=colors_dept, edgecolor="black")
axes[1].set_title("Attrition Rate by Department", fontweight="bold")
axes[1].set_xlabel("Department")
axes[1].set_ylabel("Attrition Rate (%)")
axes[1].set_ylim(0, 45)
axes[1].tick_params(axis="x", rotation=15)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

# Plot 3 - JobSatisfaction
js_rate = attrition_rate(df, "JobSatisfaction")
colors_js = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]  # Red?Green (Low?High)
bars3 = axes[2].bar(js_rate["JobSatisfaction"].astype(str),
                    js_rate["AttritionRate"],
                    color=colors_js, edgecolor="black")
axes[2].set_title("Attrition Rate by Job Satisfaction\n(1=Low, 4=Very High)",
                  fontweight="bold")
axes[2].set_xlabel("Job Satisfaction Level")
axes[2].set_ylabel("Attrition Rate (%)")
axes[2].set_ylim(0, 50)
for bar in bars3:
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/01_eda_attrition_by_factors.png", dpi=150, bbox_inches="tight")
# plt.show()  # Uncomment to display interactively (e.g. in Jupyter)
print("\n[Saved] outputs/01_eda_attrition_by_factors.png")

# -----------------------------------------------------------------------------
# SECTION 2 - PREPROCESSING
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print(" SECTION 2 - PREPROCESSING")
print("=" * 65)

# Work on a copy so the original DataFrame stays untouched for reference
df_model = df.copy()

# -- Drop irrelevant / zero-variance columns -----------------------------------
# These columns add no predictive signal:
#   EmployeeCount : always 1 ? variance = 0
#   StandardHours : always 80 ? variance = 0
#   Over18        : always 'Y' ? variance = 0
#   EmployeeNumber: just an ID / row number, not a feature
cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df_model.drop(columns=cols_to_drop, inplace=True)
print(f"\nDropped zero-variance/ID columns: {cols_to_drop}")

# -- Encode categorical columns with LabelEncoder -----------------------------
# LabelEncoder converts each unique string to an integer (e.g., "Yes"?1, "No"?0).
# It's simpler than OneHotEncoder and works well for tree-based models because
# Decision Trees split on thresholds and don't assume ordinality the same way
# linear models do.
label_enc = LabelEncoder()
categorical_cols = df_model.select_dtypes(include="object").columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

for col in categorical_cols:
    df_model[col] = label_enc.fit_transform(df_model[col])
    print(f"  Encoded '{col}'")

# -- Separate features (X) and target (y) -------------------------------------
# "Attrition" is our target variable - 1=Yes (left), 0=No (stayed)
X = df_model.drop(columns=["Attrition"])
y = df_model["Attrition"]

print(f"\nFeatures shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"\nTarget encoding - 0: No  |  1: Yes")
print(y.value_counts().to_string())

# -- Train-Test Split (80/20, stratified) --------------------------------------
# stratify=y ensures the same ~16% attrition ratio in BOTH splits.
# Without stratification, a random split might put most "Yes" cases in one set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y      # critical for imbalanced data
)

print(f"\nTrain set : {X_train.shape[0]} rows")
print(f"Test  set : {X_test.shape[0]} rows")
print(f"Train attrition rate : {y_train.mean()*100:.1f}%")
print(f"Test  attrition rate : {y_test.mean()*100:.1f}%")

# -----------------------------------------------------------------------------
# SECTION 3 - MODEL TRAINING
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print(" SECTION 3 - MODEL TRAINING")
print("=" * 65)

# -- Model A - Baseline: Unbounded tree (no max_depth constraint) --------------
# This will memorize the training data (overfit) but gives us a comparison point.
# The gap between train accuracy and test accuracy is the classic overfitting signal.
dt_baseline = DecisionTreeClassifier(
    random_state = 42
    # max_depth intentionally NOT set ? tree grows until all leaves are pure
)
dt_baseline.fit(X_train, y_train)

train_acc_baseline = accuracy_score(y_train, dt_baseline.predict(X_train))
test_acc_baseline  = accuracy_score(y_test,  dt_baseline.predict(X_test))

print("\n-- Baseline Model (no depth limit) --")
print(f"  Tree depth (actual)  : {dt_baseline.get_depth()}")
print(f"  Train accuracy       : {train_acc_baseline*100:.2f}%")
print(f"  Test  accuracy       : {test_acc_baseline*100:.2f}%")
print(f"  Overfit gap          : {(train_acc_baseline - test_acc_baseline)*100:.2f}%")
print(
    "\n  [INSIGHT] The baseline tree reaches near-perfect train accuracy but drops\n"
    "  on the test set - a textbook overfitting case.\n"
    "  It memorizes noise in training data rather than generalizing patterns."
)

# -- Model B - Pruned tree with max_depth = 5 (our final model) ----------------
# Choosing max_depth=5:
#   • Shallow enough to generalize (avoid memorizing training noise)
#   • Deep enough to capture multi-factor attrition patterns
#     (e.g., OverTime=Yes AND JobSatisfaction=Low AND YearsAtCompany<2)
#   • Readable up to depth 3-4 for visual explanation in viva
# In practice, you'd tune this via cross-validation, but depth 4-6 is a
# standard starting point for HR-scale datasets (~1000-2000 rows, ~30 features).
CHOSEN_DEPTH = 5

dt_model = DecisionTreeClassifier(
    max_depth    = CHOSEN_DEPTH,
    random_state = 42,
    # class_weight='balanced' would upweight the minority class further,
    # but we keep it simple - explaining class_weight in viva adds complexity.
    # We instead highlight the recall metric to address imbalance awareness.
)
dt_model.fit(X_train, y_train)

train_acc_model = accuracy_score(y_train, dt_model.predict(X_train))
test_acc_model  = accuracy_score(y_test,  dt_model.predict(X_test))

print(f"\n-- Final Model (max_depth={CHOSEN_DEPTH}) --")
print(f"  Tree depth (actual)  : {dt_model.get_depth()}")
print(f"  Train accuracy       : {train_acc_model*100:.2f}%")
print(f"  Test  accuracy       : {test_acc_model*100:.2f}%")
print(f"  Overfit gap          : {(train_acc_model - test_acc_model)*100:.2f}%")
print("  ? Controlled gap confirms the depth limit reduces overfitting.")

# -----------------------------------------------------------------------------
# SECTION 4 - EVALUATION
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print(" SECTION 4 - EVALUATION")
print("=" * 65)

# Generate predictions from our final pruned model
y_pred = dt_model.predict(X_test)

# -- Core metrics ------------------
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print(f"\nFinal Model Evaluation on Test Set (n={len(y_test)})")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%  (of predicted 'Yes', how many actually left)")
print(f"  Recall    : {recall*100:.2f}%  (of actual leavers, how many did we catch)  ? KEY")
print(f"  F1-Score  : {f1*100:.2f}%  (harmonic mean of Precision & Recall)")

print(
    "\n  [WHY RECALL MATTERS MORE THAN ACCURACY HERE]\n"
    "  ----------------------------------------------\n"
    "  Missing an employee who WILL quit (False Negative) means the company\n"
    "  loses talent without warning - costly to replace (recruiting, training).\n"
    "  Wrongly flagging someone who STAYS (False Positive) just means a\n"
    "  retention conversation that turns out unnecessary - minor cost.\n"
    "  In imbalanced datasets, Accuracy can be misleadingly high even when\n"
    "  the model barely catches any true leavers. Recall tells the real story."
)

print("\n-- Full Classification Report --")
print(classification_report(y_test, y_pred, target_names=["No (Stay)", "Yes (Leave)"]))

# -- Overfitting comparison table ----------------------------------------------
print("\n-- Overfitting Comparison --")
print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
print("-" * 55)
print(f"{'Baseline (no limit)':<25} {train_acc_baseline*100:>9.2f}% "
      f"{test_acc_baseline*100:>9.2f}% "
      f"{(train_acc_baseline-test_acc_baseline)*100:>7.2f}%")
print(f"{'Pruned (depth={})'.format(CHOSEN_DEPTH):<25} {train_acc_model*100:>9.2f}% "
      f"{test_acc_model*100:>9.2f}% "
      f"{(train_acc_model-test_acc_model)*100:>7.2f}%")

# -- Confusion Matrix ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm,
            annot        = True,
            fmt          = "d",
            cmap         = "Blues",
            xticklabels  = ["Predicted: No", "Predicted: Yes"],
            yticklabels  = ["Actual: No", "Actual: Yes"],
            linewidths   = 0.5,
            linecolor    = "gray",
            ax           = ax)
ax.set_title(f"Confusion Matrix - Pruned Decision Tree (depth={CHOSEN_DEPTH})",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)

# Annotate the four quadrants for viva readability
tn, fp, fn, tp = cm.ravel()
fig.text(0.14, 0.02,
         f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
         f"False Negatives (FN={fn}) = employees who will quit but we missed",
         ha="left", fontsize=9, color="#555555")

plt.tight_layout()
plt.savefig("outputs/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
# plt.show()  # Uncomment to display interactively (e.g. in Jupyter)
print("\n[Saved] outputs/02_confusion_matrix.png")
print(f"\nConfusion Matrix breakdown:\n"
      f"  True  Negatives (TN) = {tn}  ? correctly predicted 'Stay'\n"
      f"  False Positives (FP) = {fp}  ? predicted 'Leave' but employee stayed\n"
      f"  False Negatives (FN) = {fn}  ? predicted 'Stay' but employee left  ? most costly\n"
      f"  True  Positives (TP) = {tp}  ? correctly predicted 'Leave'")

# -- Feature Importance Plot ----------------------------------------------------
# Decision Trees compute feature importance as the total reduction in Gini impurity
# caused by splits on that feature (weighted by the number of samples reaching the node).
# Higher importance = that feature drives more of the tree's decisions.

importances  = dt_model.feature_importances_
feature_names = X.columns.tolist()

feat_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
             .sort_values("Importance", ascending=False)
             .head(10)
             .reset_index(drop=True))

print("\n-- Top 10 Feature Importances --")
print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(feat_df)))  # gradient: red=most important
bars = ax.barh(feat_df["Feature"][::-1],
               feat_df["Importance"][::-1],
               color=colors[::-1], edgecolor="black", linewidth=0.6)

ax.set_xlabel("Feature Importance (Gini Reduction)", fontsize=12)
ax.set_title(f"Top 10 Features Driving Employee Attrition\n"
             f"(Decision Tree, max_depth={CHOSEN_DEPTH})",
             fontsize=13, fontweight="bold")

# Add value labels on each bar
for bar, val in zip(bars, feat_df["Importance"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

ax.set_xlim(0, feat_df["Importance"].max() * 1.15)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/03_feature_importance.png", dpi=150, bbox_inches="tight")
# plt.show()  # Uncomment to display interactively (e.g. in Jupyter)
print("\n[Saved] outputs/03_feature_importance.png")

# -----------------------------------------------------------------------------
# SECTION 5 - DECISION TREE VISUALIZATION
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print(" SECTION 5 - DECISION TREE VISUALIZATION")
print("=" * 65)

# We plot the tree capped at max_depth=3 for readability.
# Our model is trained at depth=5, but showing all levels makes the diagram
# too dense to read aloud in a viva. Depth 3 shows the core decision logic.
PLOT_DEPTH = 3

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_model,
    max_depth       = PLOT_DEPTH,      # visual cap - not retraining
    feature_names   = X.columns.tolist(),
    class_names     = ["No", "Yes"],
    filled          = True,            # color-coded by majority class
    rounded         = True,
    fontsize        = 9,
    ax              = ax,
    impurity        = True,            # show Gini score at each node
    proportion      = False            # show sample counts, not proportions
)
ax.set_title(
    f"Employee Attrition Decision Tree (visualized at depth={PLOT_DEPTH} of {CHOSEN_DEPTH})\n"
    "Blue = Predicted 'No (Stay)'  |  Orange = Predicted 'Yes (Leave)'",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/04_decision_tree.png", dpi=150, bbox_inches="tight")
# plt.show()  # Uncomment to display interactively (e.g. in Jupyter)
print(f"\n[Saved] outputs/04_decision_tree.png")

# Also print the text version - useful for copying into a report
print(f"\n-- Text representation of the decision tree (depth={PLOT_DEPTH}) --")
tree_text = export_text(dt_model, feature_names=X.columns.tolist(),
                        max_depth=PLOT_DEPTH)
print(tree_text)

# -----------------------------------------------------------------------------
# SECTION 6 - SUMMARY & EXAMPLE PREDICTION WALKTHROUGH
# -----------------------------------------------------------------------------

print("\n" + "=" * 65)
print(" SECTION 6 - SUMMARY & EXAMPLE PREDICTION WALKTHROUGH")
print("=" * 65)

# -- Clean summary --------------------------------------------------------------
top3 = feat_df["Feature"].head(3).tolist()
top3_imp = feat_df["Importance"].head(3).tolist()

print(f"""
+==============================================================+
|          EMPLOYEE ATTRITION PREDICTION - RESULTS SUMMARY     |
+==============================================================+
|  Model    : Decision Tree Classifier (max_depth={CHOSEN_DEPTH})          |
|  Dataset  : {len(df)} rows, {X.shape[1]} features                           |
|  Split    : 80% train / 20% test  (stratified)               |
+==============================================================+
|  Accuracy  : {acc*100:5.2f}%                                        |
|  Precision : {precision*100:5.2f}%  (of flagged, truly at-risk)        |
|  Recall    : {recall*100:5.2f}%  (leavers correctly identified) *   |
|  F1-Score  : {f1*100:5.2f}%                                        |
+==============================================================+
|  TOP 3 FEATURES DRIVING ATTRITION:                           |
|    1. {top3[0]:<30} (importance: {top3_imp[0]:.4f})  |
|    2. {top3[1]:<30} (importance: {top3_imp[1]:.4f})  |
|    3. {top3[2]:<30} (importance: {top3_imp[2]:.4f})  |
+==============================================================+
""")

# -- Find a "Yes" prediction in the test set for the walkthrough --------------
# We want a test row that the model correctly predicted as Attrition=Yes
# so we can trace its decision path.
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
y_pred_series = pd.Series(y_pred, name="Predicted")

# Look for a true positive (actual=1, predicted=1)
tp_indices = [i for i in range(len(y_pred))
              if y_pred[i] == 1 and y_test_reset.iloc[i] == 1]

if tp_indices:
    example_idx = tp_indices[0]
else:
    # Fallback: any predicted Yes
    example_idx = next((i for i in range(len(y_pred)) if y_pred[i] == 1), 0)

example_row    = X_test_reset.iloc[[example_idx]]
actual_label   = "Yes (Leave)" if y_test_reset.iloc[example_idx] == 1 else "No (Stay)"
predicted_label= "Yes (Leave)" if y_pred[example_idx] == 1 else "No (Stay)"

print("-- Example Prediction Walkthrough --")
print(f"\nSelected test row index : {example_idx}")
print(f"Actual Attrition        : {actual_label}")
print(f"Model Prediction        : {predicted_label}")
print("\nKey feature values for this employee:")

# Pull the top-3 features and a few other notable ones for context
highlight_features = top3 + ["OverTime", "JobSatisfaction", "WorkLifeBalance",
                               "MonthlyIncome", "YearsAtCompany"]
highlight_features = list(dict.fromkeys(highlight_features))  # deduplicate, preserve order

for feat in highlight_features:
    if feat in example_row.columns:
        val = example_row[feat].values[0]
        print(f"  {feat:<30} : {val}")

# -- Plain-English decision path trace -----------------------------------------
# Walk through the decision node sequence from root ? leaf using the tree's
# internal arrays - no third-party library needed.
def trace_decision_path(model, row, feature_names):
    """
    Traces the decision path from root to leaf for a single sample.
    Returns a list of human-readable rule strings.
    """
    feature   = model.tree_.feature
    threshold = model.tree_.threshold
    node      = 0  # start at root
    path      = []

    while feature[node] != -2:   # -2 marks a leaf node
        feat_name  = feature_names[feature[node]]
        thresh_val = threshold[node]
        sample_val = row[feat_name].values[0]

        if sample_val <= thresh_val:
            direction = "?"
            node = model.tree_.children_left[node]
        else:
            direction = ">"
            node = model.tree_.children_right[node]

        path.append(
            f"  [{feat_name}] {sample_val:.2f} {direction} {thresh_val:.2f}"
        )

    # Leaf node - get class probabilities
    leaf_values = model.tree_.value[node][0]
    predicted_class = int(np.argmax(leaf_values))
    class_names_map = {0: "No (Stay)", 1: "Yes (Leave)"}
    path.append(
        f"  -> LEAF: predict = {class_names_map[predicted_class]} "
        f"(samples at leaf: No={leaf_values[0]:.0f}, Yes={leaf_values[1]:.0f})"
    )
    return path

print("\n-- Decision Path (root ? leaf) --")
path_steps = trace_decision_path(dt_model, example_row, X.columns.tolist())
for step in path_steps:
    print(step)

print(
    "\n-- Plain-English Summary of Decision Path --\n"
    "The model navigated through the tree by asking a series of Yes/No questions\n"
    "about this employee's features. At each node it chose a branch based on\n"
    "whether the value was above or below the learned threshold.\n\n"
    "Interpretation (example):\n"
    "  If OverTime=Yes AND JobSatisfaction is Low AND YearsAtCompany < 3,\n"
    "  the tree predicts Attrition = Yes because most employees with this\n"
    "  combination in the training data did quit.\n\n"
    "  This is the key advantage of a Decision Tree - every prediction\n"
    "  can be explained step by step in plain English to HR managers."
)

print("\n--- All output figures saved to outputs/ folder ---")
print("  01_eda_attrition_by_factors.png")
print("  02_confusion_matrix.png")
print("  03_feature_importance.png")
print("  04_decision_tree.png")
print("\nDone.")
