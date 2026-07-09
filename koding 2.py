
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================
# LOAD DATA
# =====================
FILE = "Dataset_Revisi_Indikator_Kecanduan.csv"
df = pd.read_csv(FILE)

print(df.head())
print(df.info())

drop_cols = ["Student_ID","Addicted_Score","Sleep_Addiction_Indicator","Physical_Activity_Indicator"]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

target = "Addiction_Level"

X = df.drop(columns=[target])
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Label encode target
y = LabelEncoder().fit_transform(y)

# Encode categorical predictors
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Naive Bayes": GaussianNB(),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append((name, acc))
    print("="*60)
    print(name)
    print("Accuracy:", acc)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

results_df = pd.DataFrame(results, columns=["Model","Accuracy"])
print(results_df.sort_values("Accuracy", ascending=False))

plt.figure(figsize=(8,4))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

rf = models["Random Forest"]
if hasattr(rf, "feature_importances_"):
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,4))
    imp.plot(kind="bar")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.show()
# ==========================================================
# EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================================

print("\n" + "="*60)
print("UKURAN DATASET")
print("="*60)

print(f"Jumlah Baris : {df.shape[0]}")
print(f"Jumlah Kolom : {df.shape[1]}")

print("\n" + "="*60)
print("TIPE DATA")
print("="*60)

print(df.dtypes)

print("\n" + "="*60)
print("MISSING VALUE")
print("="*60)

missing = df.isnull().sum()

print(missing)

plt.figure(figsize=(10,5))
missing.plot(kind='bar')
plt.title("Jumlah Missing Value")
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DATA DUPLIKAT")
print("="*60)

duplicates = df.duplicated().sum()

print(f"Jumlah Data Duplikat : {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplikat berhasil dihapus")

print("\n" + "="*60)
print("STATISTIK DESKRIPTIF")
print("="*60)

print(df.describe())

# ==========================================================
# VISUALISASI TARGET
# ==========================================================

plt.figure(figsize=(8,5))

df["Addiction_Level"].value_counts().plot(
    kind="bar"
)

plt.title("Distribusi Tingkat Kecanduan")
plt.xlabel("Addiction Level")
plt.ylabel("Jumlah")

plt.tight_layout()
plt.show()

# ==========================================================
# HISTOGRAM NUMERIK
# ==========================================================

numeric_cols = [
    col for col in df.columns
    if df[col].dtype != 'object'
]

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    plt.hist(
        df[col],
        bins=20
    )

    plt.title(f"Distribusi {col}")

    plt.tight_layout()

    plt.show()

# ==========================================================
# BOXPLOT OUTLIER
# ==========================================================

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    plt.boxplot(df[col])

    plt.title(f"Boxplot {col}")

    plt.tight_layout()

    plt.show()

# ==========================================================
# KORELASI
# ==========================================================

temp_df = df.copy()

label_temp = LabelEncoder()

for col in temp_df.columns:

    if temp_df[col].dtype == "object":

        temp_df[col] = label_temp.fit_transform(
            temp_df[col]
        )

corr_matrix = temp_df.corr()

plt.figure(figsize=(12,8))

plt.imshow(
    corr_matrix,
    cmap='coolwarm',
    aspect='auto'
)

plt.colorbar()

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.show()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

if "Avg_Daily_Usage_Hours" in df.columns:

    df["Usage_Category"] = pd.cut(

        df["Avg_Daily_Usage_Hours"],

        bins=[0,2,4,6,8,24],

        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

if "Sleep_Hours_Per_Night" in df.columns:

    df["Sleep_Category"] = pd.cut(

        df["Sleep_Hours_Per_Night"],

        bins=[0,5,7,9,15],

        labels=[
            "Kurang",
            "Normal",
            "Baik",
            "Sangat Baik"
        ]
    )

print("Feature Engineering Selesai")

# ==========================================================
# PREPROCESSING
# ==========================================================

target = "Addiction_Level"

X = df.drop(columns=[target])

y = df[target]

remove_cols = []

for c in [
    "Student_ID",
    "Addicted_Score",
    "Sleep_Addiction_Indicator",
    "Physical_Activity_Indicator"
]:

    if c in X.columns:
        remove_cols.append(c)

X = X.drop(columns=remove_cols)

print("\nKolom Digunakan:")

for col in X.columns:
    print(col)

print("\nTarget:")
print(target)
# ==========================================================
# ONE HOT ENCODING
# ==========================================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

target = "Addiction_Level"

X = df.drop(columns=[target])
y = df[target]

drop_columns = [
    "Student_ID",
    "Addicted_Score",
    "Sleep_Addiction_Indicator",
    "Physical_Activity_Indicator"
]

for col in drop_columns:
    if col in X.columns:
        X.drop(columns=col, inplace=True)

categorical_columns = X.select_dtypes(include="object").columns

print("\nKolom Kategorik")
print(categorical_columns)

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape Setelah Encoding")
print(X.shape)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nKelas Target")
print(label_encoder.classes_)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining :", X_train.shape)

print("Testing :", X_test.shape)

# ==========================================================
# STANDARD SCALER
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================================
# IMPORT MODEL
# ==========================================================

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

try:

    from xgboost import XGBClassifier

    xgb_available = True

except:

    xgb_available = False

# ==========================================================
# MEMBUAT MODEL
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(

        random_state=42

    ),

    "Random Forest": RandomForestClassifier(

        n_estimators=300,

        random_state=42

    ),

    "KNN": KNeighborsClassifier(

        n_neighbors=5

    ),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(

        probability=True,

        random_state=42

    )

}

if xgb_available:

    models["XGBoost"] = XGBClassifier(

        eval_metric="mlogloss",

        random_state=42

    )

# ==========================================================
# TRAINING
# ==========================================================

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

results = []

trained_models = {}

for name, model in models.items():

    print("\n")

    print("="*70)

    print(name)

    print("="*70)

    if name in [

        "Logistic Regression",

        "KNN",

        "SVM"

    ]:

        model.fit(

            X_train_scaled,

            y_train

        )

        prediction = model.predict(

            X_test_scaled

        )

    else:

        model.fit(

            X_train,

            y_train

        )

        prediction = model.predict(

            X_test

        )

    trained_models[name] = model

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction,

        average="weighted"

    )

    recall = recall_score(

        y_test,

        prediction,

        average="weighted"

    )

    f1 = f1_score(

        y_test,

        prediction,

        average="weighted"

    )

    results.append(

        [

            name,

            accuracy,

            precision,

            recall,

            f1

        ]

    )

    print(

        "Accuracy :",

        round(

            accuracy,

            4

        )

    )

    print(

        "Precision :",

        round(

            precision,

            4

        )

    )

    print(

        "Recall :",

        round(

            recall,

            4

        )

    )

    print(

        "F1 Score :",

        round(

            f1,

            4

        )

    )

    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            prediction

        )

    )

    print("\nConfusion Matrix")

    print(

        confusion_matrix(

            y_test,

            prediction

        )

    )

# ==========================================================
# HASIL SEMUA MODEL
# ==========================================================

hasil = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

hasil = hasil.sort_values(

    by="Accuracy",

    ascending=False

)

print("\n")

print("="*80)

print("PERBANDINGAN MODEL")

print("="*80)

print(hasil)

# ==========================================================
# GRAFIK AKURASI
# ==========================================================

plt.figure(figsize=(12,6))

plt.bar(

    hasil["Model"],

    hasil["Accuracy"]

)

plt.xticks(rotation=20)

plt.ylabel("Accuracy")

plt.title("Perbandingan Akurasi Semua Model")

for i, v in enumerate(hasil["Accuracy"]):

    plt.text(

        i,

        v,

        str(round(v,3)),

        ha="center"

    )

plt.tight_layout()

plt.show()
# ==========================================================
# CROSS VALIDATION
# ==========================================================

from sklearn.model_selection import cross_val_score

print("\n")
print("="*80)
print("CROSS VALIDATION")
print("="*80)

cv_results = []

for name, model in models.items():

    print("\n")

    print(name)

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=10,
            scoring="accuracy"
        )

    else:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=10,
            scoring="accuracy"
        )

    print(scores)

    print("Mean Accuracy :", scores.mean())

    cv_results.append([
        name,
        scores.mean(),
        scores.std()
    ])

cv_df = pd.DataFrame(

    cv_results,

    columns=[
        "Model",
        "Mean Accuracy",
        "Std"
    ]

)

print("\n")

print(cv_df)

# ==========================================================
# GRID SEARCH
# ==========================================================

from sklearn.model_selection import GridSearchCV

print("\n")
print("="*80)
print("GRID SEARCH RANDOM FOREST")
print("="*80)

param_grid = {

    "n_estimators":[
        100,
        200,
        300
    ],

    "max_depth":[
        None,
        5,
        10,
        20
    ],

    "min_samples_split":[
        2,
        5,
        10
    ],

    "min_samples_leaf":[
        1,
        2,
        4
    ]

}

grid = GridSearchCV(

    estimator=RandomForestClassifier(random_state=42),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

print("\nBest Parameter")

print(grid.best_params_)

print("\nBest Accuracy")

print(grid.best_score_)

best_rf = grid.best_estimator_

# ==========================================================
# EVALUASI RANDOM FOREST TERBAIK
# ==========================================================

pred_rf = best_rf.predict(X_test)

print("\n")

print(classification_report(

    y_test,

    pred_rf

))

print(confusion_matrix(

    y_test,

    pred_rf

))

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_rf.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*80)

print("TOP FEATURE")

print("="*80)

print(

    importance.head(20)

)

plt.figure(figsize=(12,8))

plt.barh(

    importance.head(20)["Feature"],

    importance.head(20)["Importance"]

)

plt.title(

    "Top 20 Feature Importance"

)

plt.tight_layout()

plt.show()

# ==========================================================
# ROC CURVE
# ==========================================================

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

classes = np.unique(y)

y_test_bin = label_binarize(

    y_test,

    classes=classes

)

if hasattr(best_rf,"predict_proba"):

    y_score = best_rf.predict_proba(

        X_test

    )

    plt.figure(figsize=(8,6))

    for i in range(len(classes)):

        fpr,tpr,_ = roc_curve(

            y_test_bin[:,i],

            y_score[:,i]

        )

        roc_auc = auc(

            fpr,

            tpr

        )

        plt.plot(

            fpr,

            tpr,

            label=f"Class {i} AUC={roc_auc:.2f}"

        )

    plt.plot(

        [0,1],

        [0,1],

        "--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

# ==========================================================
# SIMPAN MODEL
# ==========================================================

import joblib

joblib.dump(

    best_rf,

    "Model_Kecanduan_RF.pkl"

)

joblib.dump(

    scaler,

    "Scaler.pkl"

)

print("\nModel berhasil disimpan.")

# ==========================================================
# PREDIKSI DATA BARU
# ==========================================================

print("\n")
print("="*80)
print("CONTOH PREDIKSI DATA BARU")
print("="*80)

contoh = X.iloc[[0]]

hasil_prediksi = best_rf.predict(contoh)

hasil_label = label_encoder.inverse_transform(

    hasil_prediksi

)

print("Prediksi Tingkat Kecanduan :")

print(hasil_label)

# ==========================================================
# MENENTUKAN MODEL TERBAIK
# ==========================================================

print("\n")
print("="*80)
print("MODEL TERBAIK")
print("="*80)

best_model = hasil.iloc[0]

print(best_model)

print("\n")

print("Nama Model :",best_model["Model"])

print("Accuracy :",best_model["Accuracy"])

print("Precision :",best_model["Precision"])

print("Recall :",best_model["Recall"])

print("F1 Score :",best_model["F1 Score"])

print("\n")

print("Program selesai dijalankan.")
# ==========================================================
# FEATURE SELECTION (SELECTKBEST)
# ==========================================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selector = SelectKBest(
    score_func=chi2,
    k=10
)

X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

print("\n")
print("="*80)
print("FITUR TERPILIH")
print("="*80)

for feature in selected_features:
    print(feature)

feature_score = pd.DataFrame({

    "Feature":X.columns,

    "Score":selector.scores_

})

feature_score = feature_score.sort_values(

    by="Score",

    ascending=False

)

print(feature_score)

plt.figure(figsize=(12,6))

plt.bar(

    feature_score["Feature"][:15],

    feature_score["Score"][:15]

)

plt.xticks(rotation=90)

plt.title("Top Feature Score")

plt.tight_layout()

plt.show()

# ==========================================================
# PCA
# ==========================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

print("\nVariance Ratio")

print(pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))

plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    c=y_train,

    cmap="viridis"

)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("PCA Projection")

plt.show()

# ==========================================================
# LEARNING CURVE
# ==========================================================

from sklearn.model_selection import learning_curve

train_sizes,train_scores,test_scores = learning_curve(

    RandomForestClassifier(

        random_state=42

    ),

    X,

    y,

    cv=5,

    scoring="accuracy",

    train_sizes=np.linspace(

        0.1,

        1.0,

        10

    )

)

train_mean = np.mean(

    train_scores,

    axis=1

)

test_mean = np.mean(

    test_scores,

    axis=1

)

plt.figure(figsize=(8,6))

plt.plot(

    train_sizes,

    train_mean,

    label="Training"

)

plt.plot(

    train_sizes,

    test_mean,

    label="Validation"

)

plt.xlabel("Training Size")

plt.ylabel("Accuracy")

plt.title("Learning Curve")

plt.legend()

plt.grid()

plt.show()

# ==========================================================
# VALIDATION CURVE
# ==========================================================

from sklearn.model_selection import validation_curve

param_range = [

    10,

    50,

    100,

    150,

    200,

    300,

    500

]

train_score,val_score = validation_curve(

    RandomForestClassifier(

        random_state=42

    ),

    X,

    y,

    param_name="n_estimators",

    param_range=param_range,

    cv=5,

    scoring="accuracy"

)

train_mean = train_score.mean(axis=1)

val_mean = val_score.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(

    param_range,

    train_mean,

    marker="o",

    label="Training"

)

plt.plot(

    param_range,

    val_mean,

    marker="s",

    label="Validation"

)

plt.xlabel("Number of Trees")

plt.ylabel("Accuracy")

plt.title("Validation Curve")

plt.legend()

plt.grid()

plt.show()
# ==========================================================
# VOTING CLASSIFIER
# ==========================================================

from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(

    estimators=[

        ("lr", LogisticRegression(max_iter=1000)),

        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )),

        ("svm", SVC(
            probability=True,
            random_state=42
        ))

    ],

    voting="soft"

)

voting_model.fit(

    X_train_scaled,

    y_train

)

voting_pred = voting_model.predict(

    X_test_scaled

)

print("\n")

print("="*80)

print("VOTING CLASSIFIER")

print("="*80)

print(

    classification_report(

        y_test,

        voting_pred

    )

)

print(

    confusion_matrix(

        y_test,

        voting_pred

    )

)

print(

    "Accuracy :",

    accuracy_score(

        y_test,

        voting_pred

    )

)

# ==========================================================
# STACKING CLASSIFIER
# ==========================================================

from sklearn.ensemble import StackingClassifier

estimators = [

    ("rf",

     RandomForestClassifier(

        random_state=42

     )),

    ("knn",

     KNeighborsClassifier()),

    ("dt",

     DecisionTreeClassifier(

        random_state=42

     ))

]

stack_model = StackingClassifier(

    estimators=estimators,

    final_estimator=LogisticRegression()

)

stack_model.fit(

    X_train_scaled,

    y_train

)

stack_pred = stack_model.predict(

    X_test_scaled

)

print("\n")

print("="*80)

print("STACKING CLASSIFIER")

print("="*80)

print(

    classification_report(

        y_test,

        stack_pred

    )

)

print(

    confusion_matrix(

        y_test,

        stack_pred

    )

)

print(

    "Accuracy :",

    accuracy_score(

        y_test,

        stack_pred

    )

)

# ==========================================================
# EXPORT HASIL MODEL
# ==========================================================

hasil.to_csv(

    "Hasil_Perbandingan_Model.csv",

    index=False

)

importance.to_csv(

    "Feature_Importance.csv",

    index=False

)

print("\n")

print("File CSV berhasil dibuat.")

# ==========================================================
# EXPORT PREDIKSI
# ==========================================================

prediksi = pd.DataFrame({

    "Aktual":

    label_encoder.inverse_transform(

        y_test

    ),

    "Prediksi":

    label_encoder.inverse_transform(

        pred_rf

    )

})

prediksi.to_csv(

    "Hasil_Prediksi.csv",

    index=False

)

print("Prediksi berhasil disimpan.")

# ==========================================================
# SAVE GAMBAR
# ==========================================================

plt.figure(figsize=(10,5))

plt.bar(

    hasil["Model"],

    hasil["Accuracy"]

)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "Accuracy_Model.png",

    dpi=300

)

plt.close()

plt.figure(figsize=(10,6))

plt.barh(

    importance.head(20)["Feature"],

    importance.head(20)["Importance"]

)

plt.tight_layout()

plt.savefig(

    "Feature_Importance.png",

    dpi=300

)

plt.close()

print("Semua gambar berhasil disimpan.")

# ==========================================================
# SIMPAN SEMUA MODEL
# ==========================================================

joblib.dump(

    trained_models["Logistic Regression"],

    "LogisticRegression.pkl"

)

joblib.dump(

    trained_models["Decision Tree"],

    "DecisionTree.pkl"

)

joblib.dump(

    trained_models["Random Forest"],

    "RandomForest.pkl"

)

joblib.dump(

    trained_models["KNN"],

    "KNN.pkl"

)

joblib.dump(

    trained_models["Naive Bayes"],

    "NaiveBayes.pkl"

)

joblib.dump(

    trained_models["SVM"],

    "SVM.pkl"

)

if xgb_available:

    joblib.dump(

        trained_models["XGBoost"],

        "XGBoost.pkl"

    )

print("Semua model berhasil disimpan.")
# ==========================================================
# CONFUSION MATRIX HEATMAP
# ==========================================================

import seaborn as sns

print("\n")
print("="*80)
print("CONFUSION MATRIX HEATMAP")
print("="*80)

for name, model in trained_models.items():

    print(f"\n{name}")

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        pred = model.predict(X_test_scaled)

    else:

        pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d"
    )

    plt.xlabel("Prediksi")

    plt.ylabel("Aktual")

    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        f"{name}_ConfusionMatrix.png",
        dpi=300
    )

    plt.show()

# ==========================================================
# PRECISION RECALL CURVE
# ==========================================================

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(
    y_test,
    classes=np.unique(y)
)

if hasattr(best_rf, "predict_proba"):

    probability = best_rf.predict_proba(X_test)

    plt.figure(figsize=(8,6))

    for i in range(len(np.unique(y))):

        precision, recall, _ = precision_recall_curve(
            y_bin[:,i],
            probability[:,i]
        )

        plt.plot(
            recall,
            precision,
            label=f"Class {i}"
        )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "Precision_Recall_Curve.png",
        dpi=300
    )

    plt.show()

# ==========================================================
# PREDIKSI SATU DATA
# ==========================================================

print("\n")
print("="*80)
print("PREDIKSI SATU DATA")
print("="*80)

contoh_data = X.iloc[[0]]

hasil_prediksi = best_rf.predict(contoh_data)

hasil_prob = best_rf.predict_proba(contoh_data)

print("Label Prediksi :")

print(
    label_encoder.inverse_transform(
        hasil_prediksi
    )[0]
)

print("\nProbabilitas")

for i, kelas in enumerate(label_encoder.classes_):

    print(
        kelas,
        ":",
        round(
            hasil_prob[0][i]*100,
            2
        ),
        "%"
    )

# ==========================================================
# RANKING MODEL
# ==========================================================

ranking = hasil.sort_values(
    by="Accuracy",
    ascending=False
)

ranking["Ranking"] = range(
    1,
    len(ranking)+1
)

print("\n")
print("="*80)
print("RANKING MODEL")
print("="*80)

print(ranking)

ranking.to_csv(
    "Ranking_Model.csv",
    index=False
)

# ==========================================================
# LAPORAN OTOMATIS
# ==========================================================

with open(
    "Laporan_Hasil_Model.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("="*60+"\n")
    f.write("HASIL PENGUJIAN MODEL MACHINE LEARNING\n")
    f.write("="*60+"\n\n")

    f.write(f"Jumlah Data : {len(df)}\n")
    f.write(f"Jumlah Fitur : {X.shape[1]}\n\n")

    f.write("Ranking Model\n\n")

    f.write(ranking.to_string())

    f.write("\n\n")

    terbaik = ranking.iloc[0]

    f.write(
        f"Model Terbaik : {terbaik['Model']}\n"
    )

    f.write(
        f"Accuracy : {terbaik['Accuracy']:.4f}\n"
    )

    f.write(
        f"Precision : {terbaik['Precision']:.4f}\n"
    )

    f.write(
        f"Recall : {terbaik['Recall']:.4f}\n"
    )

    f.write(
        f"F1 Score : {terbaik['F1 Score']:.4f}\n"
    )

print("Laporan berhasil dibuat.")

# ==========================================================
# RINGKASAN HASIL
# ==========================================================

print("\n")
print("="*80)
print("RINGKASAN HASIL PENGUJIAN")
print("="*80)

print(f"Jumlah Data           : {len(df)}")
print(f"Jumlah Fitur          : {X.shape[1]}")
print(f"Jumlah Model          : {len(models)}")
print(f"Model Terbaik         : {ranking.iloc[0]['Model']}")
print(f"Akurasi Terbaik       : {ranking.iloc[0]['Accuracy']:.4f}")
print(f"Precision Terbaik     : {ranking.iloc[0]['Precision']:.4f}")
print(f"Recall Terbaik        : {ranking.iloc[0]['Recall']:.4f}")
print(f"F1 Score Terbaik      : {ranking.iloc[0]['F1 Score']:.4f}")

print("\nSemua proses selesai.")

print("""
==========================================
OUTPUT YANG BERHASIL DIBUAT
==========================================

1. Hasil_Perbandingan_Model.csv
2. Ranking_Model.csv
3. Feature_Importance.csv
4. Hasil_Prediksi.csv
5. Accuracy_Model.png
6. Feature_Importance.png
7. Confusion Matrix setiap model
8. Precision_Recall_Curve.png
9. Model (*.pkl)
10. Scaler.pkl
11. Laporan_Hasil_Model.txt

==========================================
PROGRAM SELESAI
==========================================
""")