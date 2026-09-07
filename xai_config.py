"""
XAI configuration for country-specific counterfactual explanations.

Use:
- ACTIONABLE_FEATURES for DiCE `features_to_vary`
- BLOCKED_FEATURES to prevent DiCE from suggesting unrealistic or unethical changes
- SEMI_ACTIONABLE_FEATURES for future UI/LLM discussion, but not for DiCE by default
"""

ACTIONABLE_FEATURES = {
    "Belgium": [
        "school_progress",
        "outflow_position",
        "field_of_study",
        "grade_secondary_education",
        "school_year_level",
        "household_language_dutch",
    ],

    "Italy": [
        "has_computer_at_home",
        "school_track",
        "academic_performance",
        "repeated_years",
        "absences_last_year",
    ],

    "Portugal": [
        "distance_home_school",
        "state_support",
        "responsible_person",
        "educational_track",
    ],

    "Romania": [
        "distance_home_school",
        "state_support",
        "responsible_person",
    ],
}


SEMI_ACTIONABLE_FEATURES = {
    "Belgium": [
        "residence_central_city",
        "mother_education_level",
    ],

    "Italy": [
        "language_home",
        "residence_zone",
    ],

    "Portugal": [
        "father_employment_status",
        "mother_employment_status",
    ],

    "Romania": [
        "father_employment_status",
        "mother_employment_status",
    ],
}


BLOCKED_FEATURES = {
    "Belgium": [
        "schooljaar",
        "gender",
        "belgian_nonbelgian",
        "eu_noneu",
        "age_category",
        "residence_merged_municipality_name",
        "residence_province_name",
        "target",
    ],

    "Italy": [
        "gender",
        "age_group",
        "citizenship",
        "parental_education",
        "region",
        "age",
        "foreign_background",
        "dropout",
    ],

    "Portugal": [
        "birth_country",
        "gender",
        "nacionality",
        "age",
        "education_responsible_person",
        "father_education_level",
        "mother_education_level",
        "status",
    ],

    "Romania": [
        "birth_country",
        "gender",
        "nacionality",
        "age",
        "education_responsible_person",
        "father_education_level",
        "mother_education_level",
        "status",
    ],
}


def get_actionable_features(country, model_features=None):
    """
    Return only actionable features that actually exist in the trained model.
    This avoids errors when different country datasets have different columns.
    """
    features = ACTIONABLE_FEATURES.get(country, [])

    if model_features is None:
        return features

    return [f for f in features if f in model_features]


def get_blocked_features(country, model_features=None):
    """
    Return blocked features for reference, optionally filtered by model features.
    """
    features = BLOCKED_FEATURES.get(country, [])

    if model_features is None:
        return features

    return [f for f in features if f in model_features]


def is_valid_counterfactual_change(country, feature, old_val, new_val):
    """
    Post-process DiCE suggestions.
    Reject unrealistic or harmful counterfactual changes.
    """

    country = str(country or "").lower()
    feature = str(feature)


    if old_val == new_val:
        return False

    if str(old_val) == str(new_val):
        return False

    # ---------------------------
    # Belgium-specific rules
    # ---------------------------

    if country == "belgium":

        # Do not suggest moving to lower grade
        if feature == "grade_secondary_education":
            try:
                return float(new_val) >= float(old_val)
            except Exception:
                return False

        # Do not suggest moving backwards in school year level
        if feature == "school_year_level":
            try:
                return float(new_val) >= float(old_val)
            except Exception:
                return False

        # Do not suggest worsening academic delay
        if feature == "school_progress":
            delay_order = {
                "No academic delay": 0,
                "1 year of academic delay": 1,
                "2 years of academic delay": 2,
                "More than 2 years of academic delay": 3
            }

            old_rank = delay_order.get(str(old_val))
            new_rank = delay_order.get(str(new_val))

            if old_rank is None or new_rank is None:
                return False

            return new_rank <= old_rank

    # ---------------------------
    # Italy-specific rules
    # ---------------------------

    if country == "italy":

        # Absences should only decrease
        if feature == "absences_last_year":
            try:
                return float(new_val) < float(old_val)
            except Exception:
                return False

        # Repeated years should not increase
        if feature == "repeated_years":
            try:
                return float(new_val) <= float(old_val)
            except Exception:
                return False

        # Academic performance should not get worse
        if feature == "academic_performance":
            performance_order = {
                "Low": 0,
                "Medium": 1,
                "High": 2
            }

            old_rank = performance_order.get(str(old_val))
            new_rank = performance_order.get(str(new_val))

            if old_rank is None or new_rank is None:
                return False

            return new_rank >= old_rank
        
    # ---------------------------
    # Portugal and Romania rules
    # ---------------------------

    if country in ["portugal", "romania"]:

        if feature == "distance_home_school":
            distance_order = {
                "Short": 0,
                "Moderate": 1,
                "Long": 2
            }

            old_rank = distance_order.get(str(old_val))
            new_rank = distance_order.get(str(new_val))

            if old_rank is None or new_rank is None:
                return False

            # Distance should improve or stay closer, not become longer
            return new_rank <= old_rank

        if feature == "state_support":
            # Do not suggest removing support
            if str(old_val).lower() == "yes" and str(new_val).lower() == "no":
                return False

            return True

        if feature in ["responsible_person"]:
            # Allow change of responsible person, but block empty/unknown suggestions
            blocked_values = ["", "none", "nan", "unknown"]

            if str(new_val).strip().lower() in blocked_values:
                return False

            return True

    return True