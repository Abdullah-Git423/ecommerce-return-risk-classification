"""
E-commerce Return Risk Classifier — Streamlit Demo
Loads the trained Logistic Regression pipeline (with tuned threshold 0.45)
and predicts return risk for a new order.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------
# Compatibility patch for scikit-learn version differences
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Return Risk Classifier",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------
MODEL_PATH = "final_return_risk_model.pkl"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Make sure it's in the same folder as app.py.")
        st.stop()
    bundle = joblib.load(path)
    return bundle["model"], bundle["threshold"]

model, threshold = load_model(MODEL_PATH)

# ---------------------------------------------------------------
# Title
# ---------------------------------------------------------------
st.title("📦 E-commerce Return Risk Classifier")
st.markdown(
    "Predict whether a new order is likely to be **returned**. "
    "This demo uses a Logistic Regression model trained on historical order data. "
    f"**Decision threshold:** {threshold}"
)

st.markdown("---")

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.subheader("Enter order details")

col1, col2 = st.columns(2)

with col1:
    order_value = st.number_input("Order value ($)", min_value=0.0, max_value=10000.0,
                                  value=800.0, step=10.0)
    item_count = st.number_input("Item count", min_value=1, max_value=20,
                                 value=3, step=1)
    discount_pct = st.number_input("Discount %", min_value=0.0, max_value=100.0,
                                   value=10.0, step=0.5)
    delivery_time_days = st.number_input("Delivery time (days)", min_value=0.0,
                                         max_value=30.0, value=5.0, step=0.5)

with col2:
    customer_prior_orders = st.number_input("Customer prior orders", min_value=0,
                                            max_value=50, value=5, step=1)
    customer_prior_returns = st.number_input("Customer prior returns", min_value=0,
                                             max_value=50, value=1, step=1)
    customer_return_rate = st.slider("Customer historical return rate",
                                     min_value=0.0, max_value=1.0,
                                     value=0.20, step=0.01)

st.markdown("**Categorical details**")
col3, col4 = st.columns(2)
with col3:
    category = st.selectbox("Product category",
                            ["Electronics", "Fashion", "Home", "Beauty", "Books"])
with col4:
    payment_method = st.selectbox("Payment method",
                                ["Credit Card", "Debit Card", "UPI", "Wallet", "COD"])

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("🔍 Predict Return Risk", type="primary"):

    # Validate: prior returns cannot exceed prior orders
    if customer_prior_returns > customer_prior_orders and customer_prior_orders > 0:
        st.warning("Prior returns cannot exceed prior orders. Please review inputs.")
        st.stop()

    # Build a single-row DataFrame matching training columns
    input_df = pd.DataFrame([{
        "order_value": order_value,
        "item_count": item_count,
        "discount_pct": discount_pct,
        "delivery_time_days": delivery_time_days,
        "customer_prior_orders": customer_prior_orders,
        "customer_prior_returns": customer_prior_returns,
        "customer_return_rate": customer_return_rate,
        "category": category,
        "payment_method": payment_method
    }])

    # Predict probability and apply the tuned threshold
    prob_return = model.predict_proba(input_df)[0, 1]
    prediction = int(prob_return >= threshold)

    # -----------------------------------------------------------
    # Display result
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("Prediction")

    if prediction == 1:
        st.error(f"⚠️ **High Return Risk** — probability of return: {prob_return:.2%}")
    else:
        st.success(f"✅ **Low Return Risk** — probability of return: {prob_return:.2%}")

    # Probability bar
    st.progress(min(prob_return, 1.0))
    st.caption(f"Threshold = {threshold}. Probability ≥ threshold → flagged as high risk.")

    # -----------------------------------------------------------
    # Explanation using LR coefficients
    # -----------------------------------------------------------
    st.markdown("### Why this prediction?")

    try:
        clf = model.named_steps["clf"]
        pre = model.named_steps["pre"]

        # Feature names after encoding
        num_features = ['order_value', 'item_count', 'discount_pct', 'delivery_time_days',
                        'customer_prior_orders', 'customer_prior_returns', 'customer_return_rate']
        cat_features = ['category', 'payment_method']
        ohe = pre.named_transformers_["cat"].named_steps["encoder"]
        ohe_names = list(ohe.get_feature_names_out(cat_features))
        feature_names = num_features + ohe_names

        # Transformed input
        X_trans = pre.transform(input_df)

        # Linear contribution = coefficient × feature value
        if hasattr(X_trans, "toarray"):
            X_arr = X_trans.toarray()[0]
        else:
            X_arr = np.asarray(X_trans)[0]

        contributions = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": clf.coef_[0] * X_arr
        })
        contributions["AbsContribution"] = contributions["Contribution"].abs()
        top = contributions.sort_values("AbsContribution", ascending=False).head(6)

        st.markdown("**Top contributing factors** (positive pushes toward return, negative pushes away):")
        for _, row in top.iterrows():
            arrow = "🔺" if row["Contribution"] > 0 else "🔻"
            st.markdown(f"- {arrow} `{row['Feature']}` → {row['Contribution']:+.4f}")

    except Exception as e:
        st.caption(f"(Explanation unavailable: {e})")

    # -----------------------------------------------------------
    # Show raw input for transparency
    # -----------------------------------------------------------
    with st.expander("Show input details"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.markdown("---")
st.caption(
    "Educational prototype — not for production decision-making. "
    "Model: Logistic Regression trained on 6,000 historical e-commerce orders."
)