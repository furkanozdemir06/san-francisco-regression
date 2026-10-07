import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SF Rent Predictor", page_icon="🏠", layout="wide")

LAUNDRY = {"no laundry": 0, "on-site": 1, "in-unit": 2}
PARKING = {"no parking": 0, "off-street": 1, "protected": 2, "valet": 3}


@st.cache_data
def load_data():
    df = pd.read_csv("sf_clean.csv")
    df["laundry"] = df.laundry.str[4:].map(LAUNDRY)  # "(a) in-unit" -> "in-unit" -> 2
    df["parking"] = df.parking.str[4:].map(PARKING)
    df["pets"] = df.pets.str[4:]
    df["housing_type"] = df.housing_type.str[4:]
    return df


@st.cache_resource
def train_model(df):
    data = df[df.price < df.price.quantile(0.97)]  # drop extreme prices, like the notebook
    X = pd.get_dummies(data.drop(columns="price"))
    X_train, X_test, y_train, y_test = train_test_split(X, data.price, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = (r2_score(y_test, pred), mean_squared_error(y_test, pred) ** 0.5, mean_absolute_error(y_test, pred))
    return model, X.columns, metrics, pd.DataFrame({"Actual": y_test, "Predicted": pred})


df = load_data()
model, columns, (r2, rmse, mae), results = train_model(df)

st.title("🏠 San Francisco Rent Predictor")
st.caption("Estimate the monthly rent of an apartment from its features (Craigslist listings, Oct 2020).")

tab1, tab2, tab3 = st.tabs(["Predict", "Data insights", "Model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    sqft = c1.slider("Size (sqft)", 150, 3500, 900, step=50)
    beds = c1.number_input("Bedrooms", 0, 6, 1)
    bath = c1.number_input("Bathrooms", 1.0, 4.0, 1.0, step=0.5)
    housing = c2.selectbox("Housing type", sorted(df.housing_type.unique()))
    laundry = c2.selectbox("Laundry", list(LAUNDRY), index=2)
    parking = c2.selectbox("Parking", list(PARKING))
    pets = c3.selectbox("Pets", sorted(df.pets.unique()))
    district = c3.selectbox("Neighborhood district (1-10)", list(range(1, 11)), index=6)

    if st.button("Predict rent", type="primary"):
        row = pd.get_dummies(pd.DataFrame([{
            "sqft": sqft, "beds": beds, "bath": bath, "laundry": LAUNDRY[laundry], "pets": pets,
            "housing_type": housing, "parking": PARKING[parking], "hood_district": district,
        }]))
        row = row.reindex(columns=columns, fill_value=0)
        price = model.predict(row)[0]
        m1, m2 = st.columns(2)
        m1.metric("Estimated monthly rent", f"${price:,.0f}", f"± ${rmse:,.0f} typical error", delta_color="off")
        same_beds = df[df.beds == beds].price
        if len(same_beds):
            m2.metric(f"Median rent, {int(beds)}-bedroom listings", f"${same_beds.median():,.0f}")

with tab2:
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.histogram(df, x="price", nbins=40, title="Rent distribution"), use_container_width=True)
    col2.plotly_chart(px.scatter(df, x="sqft", y="price", color="housing_type", title="Size vs rent"),
                      use_container_width=True)
    avg = df.groupby("beds").price.mean().reset_index()
    col1.plotly_chart(px.bar(avg, x="beds", y="price", title="Average rent by bedrooms"),
                      use_container_width=True)
    col2.plotly_chart(px.box(df, x="hood_district", y="price", title="Rent by neighborhood district"),
                      use_container_width=True)
    st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=".2f", color_continuous_scale="RdBu_r",
                              title="Correlation matrix"), use_container_width=True)

with tab3:
    m1, m2, m3 = st.columns(3)
    m1.metric("R² score", f"{r2:.3f}")
    m2.metric("RMSE", f"${rmse:,.0f}")
    m3.metric("MAE", f"${mae:,.0f}")
    col1, col2 = st.columns(2)
    fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs predicted rent")
    fig.add_shape(type="line", x0=results.Actual.min(), y0=results.Actual.min(),
                  x1=results.Actual.max(), y1=results.Actual.max(), line=dict(dash="dash", color="red"))
    col1.plotly_chart(fig, use_container_width=True)
    imp = pd.Series(model.feature_importances_, index=columns).sort_values()
    col2.plotly_chart(px.bar(imp, orientation="h", title="Feature importance"), use_container_width=True)