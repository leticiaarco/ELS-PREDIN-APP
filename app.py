import shutil
import importlib
import pandas as pd
import numpy as np
import os, uuid, json, zipfile, io
from datetime import datetime
from flask import send_file
import joblib
from xai_config import get_actionable_features
from flask import Flask, redirect, render_template, request, jsonify, session, url_for
from utils import (
    train_model,
    save_model_package,
    list_models,
    validate_dataset,
    explain_prediction_shap,
    generate_counterfactuals,
    generate_llm_batch_summaries,
    get_clean_base_feature_name,
    simplify_feature_name,
    humanize_intervention,
    describe_feature
)


app = Flask(__name__)


UPLOAD_DIR = "uploads"
REGISTRY_DIR = "Results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)
PREDICTION_DIR = "PredictionResults"
os.makedirs(PREDICTION_DIR, exist_ok=True)

DATASETS = {}  
ACTIVE_MODEL_ID = None

app.secret_key = os.environ.get("SECRET_KEY", "change-before-sharing")

def apply_filters(df_in: pd.DataFrame, filters: dict) -> pd.DataFrame:
    schema = session.get("schema", {})
    df_filt = df_in.copy()

    for key, values in filters.items():
        real_col = schema.get(key)

        if real_col in df_filt.columns and values:
            df_filt = df_filt[df_filt[real_col].astype(str).isin(values)]

    return df_filt




@app.route("/")
def dashboard():
    """Home page with cards"""
    return render_template("home.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route('/explore')
def explore():
    if "dataset_id" not in session or "schema" not in session:
        return redirect(url_for('upload_page'))
    return render_template("explore.html")

@app.route("/predict")
def predict_page():
    """Early Risk Prediction wizard"""
    return render_template("predict.html")

@app.route("/api/datasets/upload", methods=["POST"])
def api_upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    dataset_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    file.save(save_path)

    df_local = pd.read_csv(save_path)
    cols = df_local.columns.tolist()
    session["dataset_id"] = dataset_id

    DATASETS[dataset_id] = {
        "path": save_path,
        "filename": file.filename,
        "columns": cols
    }

    return jsonify({
        "dataset_id": dataset_id,
        "filename": file.filename,
        "columns": cols
    })

@app.route("/api/session-dataset", methods=["GET"])
def api_session_dataset():
    """Return the dataset currently in the session (uploaded via Get Started)."""
    dataset_id = session.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"dataset_id": None})
    info = DATASETS[dataset_id]
    return jsonify({
        "dataset_id": dataset_id,
        "filename": info["filename"],
        "columns": info.get("columns", [])
    })

@app.route('/save-schema', methods=['POST'])
def save_schema():

    session['schema'] = {
        'target': request.form.get('target_col'),
        'dropout_value': request.form.get('dropout_value'),
        'enroll_value': request.form.get('enroll_value')
    }

    return redirect(url_for('explore'))
    
@app.route("/api/train_model", methods=["POST"])
def api_train_model():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("dataset_id")
    model_name = payload.get("model_name")
    target_col = payload.get("target_column")
    country = payload.get("country")

    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Invalid dataset_id"}), 400
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
    if not target_col:
        return jsonify({"error": "target_column is required"}), 400

    df_local = pd.read_csv(DATASETS[dataset_id]["path"])
    errors, warnings = validate_dataset(df_local, target_col)

    if errors:
        return jsonify({
            "status": "error",
            "errors": errors,
            "warnings": warnings
        }), 400
    else:
        try:
            pipeline, metrics, features, X_train = train_model(
                df_local,
                target_col,
                model_name
            )
        except Exception as e:
            return jsonify({
                "status": "error",
                "errors": [str(e)],
                "warnings": []
            }), 400


        model_id = str(uuid.uuid4())

        metadata = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_filename": DATASETS[dataset_id]["filename"],
            "country": country,
            "target_col": target_col,
            "model_name": model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "features": features
        }

        save_model_package(REGISTRY_DIR, model_id, pipeline, metadata, metrics)

        joblib.dump(
            X_train,
            os.path.join(REGISTRY_DIR, model_id, "train_data.joblib")
        )

        return jsonify({
            "status": "success",
            "model_id": model_id,
            "metadata": metadata,
            "metrics": metrics,
            "warnings": warnings
        })


@app.route("/api/models", methods=["GET"])
def api_list_models():
    return jsonify({
        "active_model_id": ACTIVE_MODEL_ID,
        "models": list_models(REGISTRY_DIR)
    })


@app.route("/api/models/<model_id>/activate", methods=["POST"])
def api_activate_model(model_id):
    global ACTIVE_MODEL_ID
    model_dir = os.path.join(REGISTRY_DIR, model_id)
    if not os.path.isdir(model_dir):
        return jsonify({"error": "Model not found"}), 404

    ACTIVE_MODEL_ID = model_id
    return jsonify({"active_model_id": ACTIVE_MODEL_ID})

@app.route("/api/models/active_status", methods=["GET"])
def get_active_status():
    global ACTIVE_MODEL_ID
    return jsonify({"active_id": ACTIVE_MODEL_ID})

@app.route("/api/models/<model_id>/download", methods=["GET"])
def api_download_model(model_id):
    model_dir = os.path.join(REGISTRY_DIR, model_id)
    if not os.path.isdir(model_dir):
        return jsonify({"error": "Model not found"}), 404

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ["pipeline.joblib", "metadata.json", "metrics.json"]:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=f"{model_id}/{fname}")

    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{model_id}.zip"
    )

def get_active_dataset():
    dataset_id = session.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return None
    return pd.read_csv(DATASETS[dataset_id]["path"])

@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    model_path = os.path.join(REGISTRY_DIR, model_id)
    if os.path.exists(model_path):
        shutil.rmtree(model_path) # Deletes the folder and its contents
        return jsonify({"status": "success", "message": "Model deleted"})
    return jsonify({"status": "error", "message": "Model not found"}), 404


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global ACTIVE_MODEL_ID

    if not ACTIVE_MODEL_ID:
        return jsonify({"error": "Activate a model first"}), 400

    payload = request.get_json(silent=True) or {}
    students = payload.get("students")
    use_llm = payload.get("use_llm", True)

    if not isinstance(students, list):
        return jsonify({"error": "Expected a list of students"}), 400

    model_dir = os.path.join(REGISTRY_DIR, ACTIVE_MODEL_ID)
    pipeline = joblib.load(os.path.join(model_dir, "pipeline.joblib"))

    train_data_path = os.path.join(model_dir, "train_data.joblib")

    if os.path.exists(train_data_path):
        X_train = joblib.load(train_data_path)
    else:
        X_train = None

    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    features = meta.get("features", [])

    country = meta.get("country")

    features_to_vary = get_actionable_features(
        country,
        features
    )

    rows = []
   
    for student in students:
        row = {feat: student.get(feat, None) for feat in features}
        rows.append(row)

    X_batch = pd.DataFrame(rows)

    predictions = pipeline.predict(X_batch)

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_batch)[:, 1]

    shap_explanations = explain_prediction_shap(
        pipeline,
        X_batch,
        top_n=3
    )

    results = []
    dropout_items_for_llm = []
    for i, student in enumerate(students):
        pred = int(predictions[i])
        proba = float(probabilities[i]) if probabilities is not None else None

        result = student.copy()
        result["prediction"] = "Dropout" if pred == 1 else "Enroll"
        result["probability"] = proba

        # SHAP 
        explanation = shap_explanations[i]

        for j, item in enumerate(explanation, start=1):
            result[f"top_factor_{j}"] = item["feature"]
            result[f"top_factor_{j}_effect"] = item["effect"]
            result[f"top_factor_{j}_direction"] = item["direction"]

        result["explanation_summary"] = "; ".join(
            [
                f"{item['feature']} {item['direction']}"
                for item in explanation
            ]
        )

        # DICE
        counterfactuals = []

        if pred == 1 and X_train is not None:
            counterfactuals = generate_counterfactuals(
                pipeline,
                X_train,
                X_batch.iloc[[i]], 
                features_to_vary=features_to_vary,
                country=country
            )

        for k, cf in enumerate(counterfactuals[:3], start=1):
            result[f"cf_feature_{k}"] = cf["feature"]
            result[f"cf_current_{k}"] = cf["current"]
            result[f"cf_suggested_{k}"] = cf["suggested"]

        if pred == 0:
            result["counterfactual_summary"] = "Not needed for Enroll prediction"
        elif counterfactuals:
            result["counterfactual_summary"] = "; ".join([
                humanize_intervention(
                    cf["feature"],
                    cf["current"],
                    cf["suggested"]
                )
                for cf in counterfactuals
            ])
        else:
            result["counterfactual_summary"] = "No clear intervention could be identified based on the available actionable factors."
        
         # ---------------------------
        # Prepare dropout rows for ONE batch LLM call
        # ---------------------------
        if pred == 1:
            unique_factors = []
            seen_factor_names = set()

            for item in explanation:
                base_name = get_clean_base_feature_name(item["feature"])

                if base_name in seen_factor_names:
                    continue

                seen_factor_names.add(base_name)

                unique_factors.append({
                    "feature": base_name,
                    "direction": item["direction"]
                })

                if len(unique_factors) == 3:
                    break

            dropout_items_for_llm.append({
                "id": str(i),
                "prediction": result["prediction"],
                "probability": result["probability"],
                "main_factors": unique_factors,
                "counterfactuals": counterfactuals[:3],
                "counterfactual_status": result.get("counterfactual_status", ""),
                "counterfactual_summary": result.get("counterfactual_summary", "")
            })
        else:
            result["llm_risk_summary"] = (
                "The student is predicted as low dropout risk."
            )
            result["llm_intervention_summary"] = (
                "No intervention is required based on the current prediction."
            )

        results.append(result)

    # ---------------------------
    # LLM summaries (or rule-based fallback)
    # ---------------------------
    if use_llm:
        llm_summaries = generate_llm_batch_summaries(dropout_items_for_llm)
    else:
        llm_summaries = {}

    for item in dropout_items_for_llm:
        row_index = int(item["id"])
        key = str(row_index)

        if use_llm:
            summary = llm_summaries.get(key, {})
            prob_pct = round(item["probability"] * 100, 1) if item["probability"] is not None else "N/A"
            factor_descriptions = []
            for f in item["main_factors"]:
                info = describe_feature(f["feature"])
                if info["description"]:
                    factor_descriptions.append(f"{info['name']} ({info['description']})")
                else:
                    factor_descriptions.append(info["name"])
            fallback_factors_text = "; ".join(factor_descriptions) or "the available risk indicators"
            fallback_risk = (
                f"The student has an estimated {prob_pct}% dropout probability. "
                f"The strongest contributing signals in this prediction relate to: {fallback_factors_text}."
            )
            results[row_index]["llm_risk_summary"] = summary.get("risk_summary", fallback_risk)
            results[row_index]["llm_intervention_summary"] = summary.get(
                "intervention_summary",
                results[row_index].get(
                    "counterfactual_summary",
                    "No clear intervention summary was generated."
                )
            )
        else:
            prob_pct = round(item["probability"] * 100, 1) if item["probability"] is not None else "N/A"
            factor_descriptions = []
            for f in item["main_factors"]:
                info = describe_feature(f["feature"])
                if info["description"]:
                    factor_descriptions.append(f"{info['name']} ({info['description']})")
                else:
                    factor_descriptions.append(info["name"])
            factors_text = "; ".join(factor_descriptions) or "the available risk indicators"
            results[row_index]["llm_risk_summary"] = (
                f"The student has an estimated {prob_pct}% dropout probability. "
                f"The strongest contributing signals in this prediction relate to: {factors_text}."
            )
            results[row_index]["llm_intervention_summary"] = results[row_index].get(
                "counterfactual_summary",
                "No clear intervention could be identified based on the available actionable factors."
            )


    batch_id = str(uuid.uuid4())
    batch_dir = os.path.join(PREDICTION_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    technical_df = pd.DataFrame(results)
    technical_path = os.path.join(batch_dir, "technical_predictions.csv")
    technical_df.to_csv(technical_path, index=False)

    summary_columns = [
        "prediction",
        "probability",
        "llm_risk_summary",
        "llm_intervention_summary",
        "counterfactual_summary"
    ]

    existing_columns = [
        col for col in summary_columns
        if col in technical_df.columns
    ]

    summary_df = technical_df[existing_columns].copy()
    summary_df.insert(0, "student_row", range(1, len(summary_df) + 1))

    summary_path = os.path.join(batch_dir, "summary_predictions.csv")
    summary_df.to_csv(summary_path, index=False)

    summary_records = summary_df.to_dict(orient="records")

    return jsonify({
        "status": "success",
        "batch_id": batch_id,
        "summary": summary_records,
        "summary_csv_url": f"/api/download_predictions/{batch_id}/summary",
        "technical_csv_url": f"/api/download_predictions/{batch_id}/technical"
    })


# -------------------------------------------Explore Page------------------------------------------------------------------
# ---------------------------
#   METADATA FOR FRONTEND
# ---------------------------

@app.route("/api/explore_metadata", methods=["GET"])
def api_explore_metadata():
    df = get_active_dataset()
    schema = session.get("schema", {})

    if df is None or not schema:
        return jsonify({"error": "No dataset/schema"}), 400

    return jsonify({
        "filter_options": {},
        "x_axis_columns": list(df.columns)
    })

# ---------------------------
#     DATA API ENDPOINTS
# ---------------------------

@app.route("/api/kpis", methods=["GET", "POST"])
def api_kpis():
    data = request.get_json(silent=True) or {}
    filters = data.get("filters", {})

    df = get_active_dataset()
    schema = session.get("schema", {})

    if df is None or not schema:
        return jsonify({"error": "No dataset or schema"}), 400

    TARGET_COL = schema.get("target")

    df_f = apply_filters(df, filters)

    total = len(df_f)
    dropout_value = schema.get("dropout_value")
    enroll_value = schema.get("enroll_value")

    dropout_count = (
        df_f[TARGET_COL].astype(str) == str(dropout_value)
    ).sum()

    enroll_count = (
        df_f[TARGET_COL].astype(str) == str(enroll_value)
    ).sum()

    dropout_pct = (dropout_count / total * 100) if total > 0 else 0
    enroll_pct = (enroll_count / total * 100) if total > 0 else 0

    return jsonify({
        "total_students": int(total),
        "dropout_count": int(dropout_count),
        "enroll_count": int(enroll_count),
        "dropout_pct": round(dropout_pct, 1),
        "enroll_pct": round(enroll_pct, 1)
    })


@app.route("/api/custom_chart", methods=["POST"])
def api_custom_chart():
    payload = request.get_json(silent=True) or {}
    x_axis = payload.get("x_axis")
    filters = payload.get("filters", {})

    df = get_active_dataset()
    schema = session.get("schema", {})

    if df is None or not schema:
        return jsonify({"error": "No dataset or schema"}), 400

    TARGET_COL = schema.get("target")
    df_f = apply_filters(df, filters)

    if x_axis not in df_f.columns:
        return jsonify({"error": f"Column {x_axis} not found"}), 400

    if df_f.empty:
        return jsonify({"x": [], "dropout": [], "enroll": []})

    grouped = df_f.groupby([x_axis, TARGET_COL]).size().unstack(fill_value=0)
    
    dropout_value = str(schema.get("dropout_value"))
    enroll_value = str(schema.get("enroll_value"))
    grouped.columns = grouped.columns.astype(str)

    if dropout_value not in grouped.columns: grouped[dropout_value] = 0
    if enroll_value not in grouped.columns: grouped[enroll_value] = 0

    return jsonify({
        "x": grouped.index.astype(str).tolist(),
        "dropout": grouped[dropout_value].tolist(),
        "enroll": grouped[enroll_value].tolist()
    })


@app.route("/api/top_risk_segments", methods=["GET", "POST"])
def api_top_risk_segments():
    """Find top N segments with highest dropout rates across all columns"""
    data = request.get_json(silent=True) or {}
    filters = data.get("filters", {})
    top_n = data.get("top_n", 5)

    df = get_active_dataset()
    schema = session.get("schema", {})
    TARGET_COL = schema.get("target")

    if df is None or not schema:
        return jsonify({"error": "No dataset or schema"}), 400

    dropout_value = str(schema.get("dropout_value")).strip()
    enroll_value = str(schema.get("enroll_value")).strip()

    df_f = apply_filters(df, filters)

    risk_segments = []

    exclude_cols = {
        TARGET_COL,
        "belgian_nonbelgian",
        "eu_noneu"
    }

    for col in df_f.columns:
        if col in exclude_cols:
            continue

        if df_f[col].dtype == "object" and len(df_f[col].unique()) > 50:
            continue

        if df_f.empty:
            continue

        grouped = (
            df_f.groupby(col)[TARGET_COL]
            .value_counts()
            .unstack(fill_value=0)
        )

        grouped.columns = grouped.columns.astype(str)

        if dropout_value not in grouped.columns:
            grouped[dropout_value] = 0

        if enroll_value not in grouped.columns:
            grouped[enroll_value] = 0

        grouped["total"] = grouped[[dropout_value, enroll_value]].sum(axis=1)

        grouped["dropout_rate"] = (
            grouped[dropout_value] / grouped["total"] * 100
        ).fillna(0)

        for idx, row in grouped.iterrows():
            risk_segments.append({
                "segment": f"{col}: {idx}",
                "column": col,
                "value": str(idx),
                "dropout_rate": round(float(row["dropout_rate"]), 1),
                "dropout_count": int(row[dropout_value]),
                "total_count": int(row["total"])
            })

    risk_segments.sort(
        key=lambda x: x["dropout_rate"],
        reverse=True
    )

    return jsonify({
        "segments": risk_segments[:top_n]
    })


@app.route("/api/distribution_overview", methods=["POST"])
def api_distribution_overview():
    """Get dropout distribution for a selected feature"""
    data = request.get_json(silent=True) or {}
    feature = data.get("feature")
    filters = data.get("filters", {})

    df = get_active_dataset()
    schema = session.get("schema", {})
    TARGET_COL = schema.get("target")

    if df is None or not schema or feature not in df.columns:
        return jsonify({"error": "Invalid feature"}), 400

    dropout_value = str(schema.get("dropout_value")).strip()
    enroll_value = str(schema.get("enroll_value")).strip()

    df_f = apply_filters(df, filters)

    if df_f.empty:
        return jsonify({"labels": [], "dropout": [], "enroll": [], "rates": []})

    grouped = (
        df_f.groupby(feature)[TARGET_COL]
        .value_counts()
        .unstack(fill_value=0)
    )

    grouped.columns = grouped.columns.astype(str)

    if dropout_value not in grouped.columns:
        grouped[dropout_value] = 0

    if enroll_value not in grouped.columns:
        grouped[enroll_value] = 0

    grouped["total"] = grouped[[dropout_value, enroll_value]].sum(axis=1)

    grouped["dropout_rate"] = (
        grouped[dropout_value] / grouped["total"] * 100
    ).fillna(0)

    return jsonify({
        "labels": grouped.index.tolist(),
        "dropout": grouped[dropout_value].tolist(),
        "enroll": grouped[enroll_value].tolist(),
        "rates": grouped["dropout_rate"].round(1).tolist()
    })


@app.route("/api/contribution_chart", methods=["GET", "POST"])
def api_contribution_chart():
    """Top N categories by dropout rate within that specific field"""
    data = request.get_json(silent=True) or {}
    filters = data.get("filters", {})
    field_key = data.get("field", "field")
    top_n = data.get("top_n", 5)

    df = get_active_dataset()
    schema = session.get("schema", {})
    TARGET_COL = schema.get("target")

    if df is None or not schema:
        return jsonify({"error": "No dataset or schema"}), 400

    dropout_value = str(schema.get("dropout_value")).strip()
    enroll_value = str(schema.get("enroll_value")).strip()
    field_col = schema.get(field_key)

    if not field_col or field_col not in df.columns:
        return jsonify({"fields": [], "dropout": [], "contribution_pct": [], "unavailable": True})

    df_f = apply_filters(df, filters)

    if df_f.empty:
        return jsonify({"fields": [], "dropout": [], "contribution_pct": [], "unavailable": True})

    # Group by field and target to get counts for each status
    grouped = (
        df_f.groupby(field_col)[TARGET_COL]
        .value_counts()
        .unstack(fill_value=0)
    )

    grouped.columns = grouped.columns.astype(str)
    if dropout_value not in grouped.columns: grouped[dropout_value] = 0
    if enroll_value not in grouped.columns: grouped[enroll_value] = 0

    # Calculate total within this specific field to avoid bias
    grouped["field_total"] = grouped[dropout_value] + grouped[enroll_value]
    
    # Calculate percentage within the field
    grouped["contribution_pct"] = (
        grouped[dropout_value] / grouped["field_total"] * 100
    ).fillna(0).round(1)

    # Sort by the internal field rate
    grouped = grouped.sort_values("contribution_pct", ascending=True).tail(top_n)

    return jsonify({
        "fields": grouped.index.tolist(),
        "dropout": grouped[dropout_value].tolist(),
        "contribution_pct": grouped["contribution_pct"].tolist()
    })

# ---------------------------
#       TRAINING ROUTE
# ---------------------------

@app.route("/train", methods=["GET", "POST"])
def train():
    """Training page for model selection"""
    accuracy = None
    selected_model = None

    if request.method == "POST":
        selected_model = request.form.get("model")
        func_path = MODEL_FUNCTIONS[selected_model]
        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        accuracy = func()

    return render_template("train.html", accuracy=accuracy, selected_model=selected_model)


@app.route("/api/target-values", methods=["POST"])
def get_target_values():
    data = request.get_json()

    dataset_id = data.get("dataset_id")
    target_col = data.get("target_col")

    if dataset_id not in DATASETS:
        return jsonify({"error": "Invalid dataset"}), 400

    df = pd.read_csv(DATASETS[dataset_id]["path"])

    if target_col not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    values = (
        df[target_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return jsonify({"values": values})



# Predictions 

@app.route("/api/download_predictions/<batch_id>/<file_type>", methods=["GET"])
def download_predictions(batch_id, file_type):
    batch_dir = os.path.join(PREDICTION_DIR, batch_id)

    if file_type == "summary":
        filename = "summary_predictions.csv"
    elif file_type == "technical":
        filename = "technical_predictions.csv"
    else:
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(batch_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "Prediction file not found"}), 404

    return send_file(
        file_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)



