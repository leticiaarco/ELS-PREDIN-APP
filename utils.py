import os, json
import joblib
import pandas as pd
import shap
import numpy as np 
import requests
import dice_ml
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xai_config import is_valid_counterfactual_change
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ---------------------------
#   FEATURE METADATA (for richer LLM explanations)
# ---------------------------

_FEATURE_METADATA_CACHE = None
_FEATURE_METADATA_PATH = os.path.join(os.path.dirname(__file__), "Meta_data.json")


def _normalize_feature_key(name):
    """Normalize a feature name/id for fuzzy lookup."""
    if name is None:
        return ""
    s = str(name).lower()
    # strip preprocessor prefixes and '= value' suffix
    s = s.replace("num__", "").replace("cat__", "")
    if "=" in s:
        s = s.split("=")[0]
    # keep only alphanumerics
    return "".join(ch for ch in s if ch.isalnum())


def load_feature_metadata(path=None):

    global _FEATURE_METADATA_CACHE
    if _FEATURE_METADATA_CACHE is not None and path is None:
        return _FEATURE_METADATA_CACHE

    meta_path = path or _FEATURE_METADATA_PATH
    lookup = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        features = (raw.get("dataset") or {}).get("predictive_features") or []
        for feat in features:
            entry = {
                "feature_id": feat.get("feature_id", ""),
                "feature_name": feat.get("feature_name", ""),
                "feature_description": feat.get("feature_description", ""),
                "feature_domain": feat.get("feature_domain", ""),
                "categorical_values": feat.get("categorical_values") or [],
            }
            for key in (feat.get("feature_id"), feat.get("feature_name")):
                norm = _normalize_feature_key(key)
                if norm:
                    lookup[norm] = entry
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        lookup = {}

    if path is None:
        _FEATURE_METADATA_CACHE = lookup
    return lookup


def describe_feature(name):

    meta = load_feature_metadata()
    entry = meta.get(_normalize_feature_key(name))
    if entry:
        return {
            "name": entry["feature_name"] or simplify_feature_name(name),
            "description": entry["feature_description"] or "",
        }
    return {
        "name": simplify_feature_name(name),
        "description": "",
    }


# ---------------------------
#   PREPROCESSING
# ---------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )


# ---------------------------
#   MODEL FACTORY
# ---------------------------

def make_model(model_name: str, params: dict | None = None):
    name = (model_name or "").lower()
    params = params or {}  # user preferences (if any)

    if name == "logistic_regression":
        defaults = {
            "max_iter": 2000,
            "C": 1.0,
            "solver": "lbfgs"
        }
        return LogisticRegression(**{**defaults, **params})

    if name == "decision_tree":
        defaults = {
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        }
        return DecisionTreeClassifier(**{**defaults, **params})

    if name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        }
        return RandomForestClassifier(**{**defaults, **params})

    if name == "neural_network":
        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_iter": 500
        }
        return MLPClassifier(**{**defaults, **params})

    if name == "xgboost":
        defaults = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42
        }
        return XGBClassifier(**{**defaults, **params})

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------
#   TRAINING LOGIC
# ---------------------------

def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    balance: bool = False
):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop constant columns (only one unique value — no predictive value)
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    y = y.apply(lambda x: 1 if str(x).lower() in ['dropout', 'inactive', 'yes'] else 0)

    # Class imbalance, step 1: keep the same class ratio in train and test
    # so the minority (dropout) class is present on both sides of the split.
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    preprocessor = build_preprocessor(X_train)
    model = make_model(model_name)

    # Class imbalance,  balance the training data with SMOTE so the

    minority_count = int(y_train.value_counts().min())
    use_smote = balance and y_train.nunique() > 1 and minority_count >= 2

    if use_smote:
        k_neighbors = min(5, minority_count - 1)
        pipeline = ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("model", model)
        ])
    else:
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return pipeline, metrics, X.columns.tolist(), X_train


# ---------------------------
#   MODEL REGISTRY
# ---------------------------

def save_model_package(registry_dir, model_id, pipeline, metadata, metrics):
    model_dir = os.path.join(registry_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(model_dir, "pipeline.joblib"))

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model_dir


def list_models(registry_dir):
    models = []
    if not os.path.exists(registry_dir):
        return models

    for mid in os.listdir(registry_dir):
        mdir = os.path.join(registry_dir, mid)
        if not os.path.isdir(mdir):
            continue

        with open(os.path.join(mdir, "metadata.json")) as f:
            metadata = json.load(f)
        with open(os.path.join(mdir, "metrics.json")) as f:
            metrics = json.load(f)

        models.append({
            "model_id": mid,
            "metadata": metadata,
            "metrics": metrics
        })
    return models


def validate_dataset(df, target_col):
    errors = []
    warnings = []

    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset.")

    if df.empty:
        errors.append("Dataset is empty.")

    if target_col in df.columns:
        y = df[target_col]
        if y.nunique() < 2:
            errors.append("Target column contains only one class.")

    missing_ratio = df.isnull().mean()
    bad_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
    if bad_cols:
        warnings.append(
            "Columns with more than 50% missing values: " + ", ".join(bad_cols)
        )

    return errors, warnings


# ---------------------------
#   Explanable AI
# ---------------------------


def clean_shap_feature_name(name):
    name = str(name)

    if name.startswith("num__"):
        return name.replace("num__", "").replace("_", " ").title()

    if name.startswith("cat__"):
        cleaned = name.replace("cat__", "")

        if "_" in cleaned:
            feature, value = cleaned.rsplit("_", 1)
            feature = feature.replace("_", " ").title()
            value = value.replace("_", " ")
            return f"{feature} = {value}"

        return cleaned.replace("_", " ").title()

    return name.replace("_", " ").title()


def get_clean_base_feature_name(feature_name):
    """
    Converts cleaned SHAP names like:
    """
    feature_name = str(feature_name)

    if "=" in feature_name:
        return feature_name.split("=")[0].strip()

    return feature_name.strip()


def simplify_feature_name(feature_name):
    """
    Converts technical names to readable names.
    """
    feature_name = str(feature_name)

    feature_name = feature_name.replace("num__", "")
    feature_name = feature_name.replace("cat__", "")
    feature_name = feature_name.replace("_", " ")
    feature_name = feature_name.replace("=", " = ")

    return " ".join(feature_name.split()).title()




def humanize_intervention(feature, current, suggested):
    """
    Convert a feature flagged by the counterfactual engine into a
    GENERAL supportive action for educators.

    Important: per consortium feedback, we never tell teachers to
    *change* a feature value (e.g. "change Educational Track from
    General to Technical"). Instead we recommend a general action
    that is relevant to that feature. The current/suggested values
    are intentionally ignored.
    """
    feature_clean = simplify_feature_name(feature).lower().strip()

    # Curated, feature-based general recommendations.
    GENERAL_ACTIONS = {
        # Family / support context
        "responsible person":
            "Explore whether additional family or guardian involvement could strengthen the student's support network",
        "education responsible person":
            "Consider involving an additional educational mentor or support contact to follow the student's progress",
        "father employment status":
            "Be mindful of possible socio-economic pressures at home and connect the family with available support services if appropriate",
        "mother employment status":
            "Be mindful of possible socio-economic pressures at home and connect the family with available support services if appropriate",
        "state support":
            "Review whether the student and family are aware of, and benefiting from, all available social or educational support schemes",
        "household language dutch":
            "Consider providing language support or peer-tutoring resources to help the student engage fully with the curriculum",
        "language home":
            "Consider providing language support or peer-tutoring resources to help the student engage fully with the curriculum",

        # Travel / access
        "distance home school":
            "Look into ways to reduce the burden of travel, for example transport assistance or schedule adjustments",
        "residence central city":
            "Be aware that students living further from the school may need extra logistical support",
        "residence zone":
            "Be aware that students living in less-served areas may need extra logistical or transport support",

        # Academic
        "academic performance":
            "Offer targeted academic support, such as tutoring or study guidance, to help reinforce the student's learning",
        "absences last year":
            "Monitor attendance closely and explore the reasons behind any recurring absences with the student and family",
        "repeated years":
            "Provide additional academic mentoring and motivational support to help the student stay on track",
        "school progress":
            "Offer structured catch-up support to help the student close any learning gaps at their own pace",
        "grade secondary education":
            "Monitor the student's academic trajectory and arrange regular check-ins to discuss progress and difficulties",
        "school year level":
            "Monitor the student's academic trajectory and arrange regular check-ins to discuss progress and difficulties",
        "school track":
            "Have an open conversation with the student and family about their educational interests and aspirations, and review whether the current programme aligns with them",
        "educational track":
            "Have an open conversation with the student and family about their educational interests and aspirations, and review whether the current programme aligns with them",
        "field of study":
            "Have an open conversation with the student about their interests and motivation to ensure they feel engaged with their studies",
        "outflow position":
            "Monitor the student's engagement and provide guidance on next-step options to keep their educational pathway open",
        "has computer at home":
            "Check whether the student has reliable access to learning materials and digital tools, and connect them with resources if needed",
    }

    if feature_clean in GENERAL_ACTIONS:
        return GENERAL_ACTIONS[feature_clean]

    # Generic, value-free fallback
    return (
        f"pay closer attention to the student's situation regarding "
        f"{simplify_feature_name(feature).lower()} and offer relevant support where possible"
    )


def explain_prediction_shap(pipeline, X_raw, top_n=3):
    """
    Generate simple SHAP explanation for one or many prediction rows.
    """

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(X_raw)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Use TreeExplainer for tree-based models (XGBoost, RandomForest, etc.)

    tree_model_names = (
        "XGBClassifier", "XGBRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
    )
    if type(model).__name__ in tree_model_names:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_processed)
    else:
        # Non-tree models (e.g. MLPClassifier) aren't callable themselves;

        predict_fn = getattr(model, "predict_proba", None) or model.predict
        background = shap.sample(X_processed, min(100, X_processed.shape[0]))
        explainer = shap.Explainer(predict_fn, background)
        shap_values = explainer(X_processed)

    values = shap_values.values

    if len(values.shape) == 3:
        values = values[:, :, 1]

    explanations = []

    for row_values in values:
        top_indices = np.argsort(np.abs(row_values))[::-1][:top_n]

        row_explanation = []
        for idx in top_indices:
            effect = float(row_values[idx])
            row_explanation.append({
                "feature": clean_shap_feature_name(feature_names[idx]),
                "effect": round(effect, 4),
                "direction": "increases dropout risk" if effect > 0 else "decreases dropout risk"
            })

        explanations.append(row_explanation)

    return explanations


def generate_counterfactuals(
    pipeline,
    X_train,
    input_row,
    target_name="target",
    total_CFs=1,
    features_to_vary=None,
    country=None
):
    """
    Generate DiCE counterfactual explanations
    """

    try:
        model = pipeline.named_steps["model"]

        # Add fake target column for DiCE schema
        train_df = X_train.copy()
        train_df[target_name] = 0

        continuous_features = (
            X_train.select_dtypes(include=["number"])
            .columns
            .tolist()
        )
        
        input_row = input_row.copy()

        for col in input_row.columns:
            if input_row[col].isnull().any():
                if col in X_train.columns:
                    if pd.api.types.is_numeric_dtype(X_train[col]):
                        input_row[col] = input_row[col].fillna(X_train[col].median())
                    else:
                        mode_value = X_train[col].mode()
                        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
                        input_row[col] = input_row[col].fillna(fill_value)

        input_row = input_row.copy()

        for col in input_row.columns:
            if col not in X_train.columns:
                continue

            # Missing values
            if input_row[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    input_row[col] = input_row[col].fillna(X_train[col].median())
                else:
                    mode_value = X_train[col].mode()
                    fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
                    input_row[col] = input_row[col].fillna(fill_value)

            # Unseen categorical values
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                allowed_values = set(X_train[col].dropna().astype(str).unique())
                mode_value = X_train[col].mode()
                fallback_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"

                input_row[col] = input_row[col].apply(
                    lambda x: x if str(x) in allowed_values else fallback_value
                )

        data_dice = dice_ml.Data(
            dataframe=train_df,
            continuous_features=continuous_features,
            outcome_name=target_name
        )

        model_dice = dice_ml.Model(
            model=pipeline,
            backend="sklearn"
        )

        exp = dice_ml.Dice(
            data_dice,
            model_dice,
            method="random"
        )

        cf = exp.generate_counterfactuals(
            input_row,
            total_CFs=total_CFs,
            desired_class="opposite",
            features_to_vary=features_to_vary
        )

        cf_df = cf.cf_examples_list[0].final_cfs_df

        if cf_df is None or cf_df.empty:
            return []

        suggestions = []

        original = input_row.iloc[0]

        for col in input_row.columns:
            old_val = original[col]
            new_val = cf_df.iloc[0][col]

            if is_valid_counterfactual_change(country, col, old_val, new_val):
                suggestions.append({
                    "feature": col,
                    "current": old_val,
                    "suggested": new_val
                })

        return suggestions

    except Exception as e:
        print("DiCE error:", e)
        return []
    

#LLM

def generate_llm_batch_summaries(dropout_items, model_name="qwen2.5:3b", batch_size=2):
    """
    Sends dropout explanations to local Ollama in small batches.
    """

    if not dropout_items:
        return {}

    safe_items = []
    for item in dropout_items:
        enriched_factors = []
        for f in (item.get("main_factors") or []):
            info = describe_feature(f.get("feature", ""))
            enriched_factors.append({
                "name": info["name"],
                "description": info["description"],
                "direction": f.get("direction", ""),
            })

        cf_features = []
        for cf in (item.get("counterfactuals") or []):
            info = describe_feature(cf.get("feature", ""))
            cf_features.append({
                "name": info["name"],
                "description": info["description"],
            })

        safe_items.append({
            "id": item.get("id"),
            "prediction": item.get("prediction"),
            "probability": item.get("probability"),
            "main_factors": enriched_factors,
            "intervention_focus_features": cf_features,
            "general_intervention_hint": item.get("counterfactual_summary", "")
        })

    all_summaries = {}

    for start in range(0, len(safe_items), batch_size):
        batch = safe_items[start:start + batch_size]

        prompt = f"""
You are helping educational staff understand student dropout risk predictions.

Each student record contains:
- "main_factors": the factors that pushed the prediction. Each one has
  a human-readable "name", a "description" explaining what the factor
  actually measures, and a "direction" (the role it played in the risk).
- "intervention_focus_features": areas the model identified as
  potentially useful to focus support on. Each one has a "name" and a
  "description".
- "general_intervention_hint": a pre-written, value-free suggestion you
  can take inspiration from.

Rules for the risk_summary:
- Use simple, clear, supportive language for teachers and counsellors.
- Always express the probability as a percentage.
- DO NOT just list factor names. For each main factor, briefly explain
  in plain words WHY it matters, using its "description" as the basis.
  For example, instead of "the student is high risk because of
  risk_home_language and mother_education", write something like
  "the language spoken at home has been flagged by the school as a
  risk indicator, and the mother's education level is also associated
  with higher dropout risk in this dataset".
- Never write generic placeholders like "based on the listed risk
  factors" or "due to various factors". Always ground the explanation
  in the actual factor descriptions.
- Do not invent information beyond what is provided.
- Do not mention SHAP, DiCE, machine learning, counterfactuals, model
  internals, row ids, or any technical identifier.
- Refer to the learner as "the student".

Rules for the intervention_summary:
- NEVER recommend changing the value of a feature. Do NOT say things
  like "change Educational Track from General to Technical",
  "change responsible person from mother to father", or
  "reduce distance from Long to Short".
- NEVER mention specific current or target values for any feature.
- Instead, propose GENERAL supportive actions that are RELEVANT to the
  flagged intervention_focus_features, using their descriptions to
  decide what kind of action makes sense. Examples of acceptable
  phrasing:
    * "Offer additional academic mentoring and monitor progress closely."
    * "Have a supportive conversation with the student and family about
       motivation and educational interests."
    * "Explore whether transport or logistical support could ease the
       student's daily school routine."
    * "Check whether the family is aware of available social or
       educational support schemes."
- Frame actions as things the school, teacher or counsellor can do.
- If no clear focus areas are provided, give a generic supportive
  recommendation (regular check-ins, mentoring, family engagement).

Output rules:
- Return ONLY valid JSON. No markdown, no comments, no trailing commas.
  Escape quotation marks correctly.
- JSON keys must be the exact student id values from the input.

JSON format:
{{
  "0": {{
    "risk_summary": "...",
    "intervention_summary": "..."
  }}
}}

Students:
{json.dumps(batch, indent=2, default=str)}
"""

        try:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
            response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 200,
                        "num_ctx": 2048,
                    }
                },
                timeout=40
            )

            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()

            start_json = raw_text.find("{")
            end_json = raw_text.rfind("}") + 1

            if start_json == -1 or end_json == -1:
                continue

            json_text = raw_text[start_json:end_json]

            try:
                batch_summaries = json.loads(json_text)
            except json.JSONDecodeError:
                print("Invalid LLM JSON output:")
                print(raw_text)
                batch_summaries = {}

            all_summaries.update(batch_summaries)

        except Exception as e:
            print("LLM batch error:", e)
            continue

    return all_summaries