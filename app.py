
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon=None,
    layout="wide"
)


DATA_PATH = "Customer_Feedback_Cleaned.csv"
ASPECT_PATH = "Aspect_Sentiment_Records.csv"



@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8")

    data["Rate"] = pd.to_numeric(
        data["Rate"],
        errors="coerce"
    )

    data["Sentiment"] = data["Rate"].apply(
        lambda rating: (
            "Positive" if pd.notna(rating) and rating >= 4
            else "Neutral" if pd.notna(rating) and rating == 3
            else "Negative" if pd.notna(rating) and rating < 3
            else "Unknown"
        )
    )

    return data



st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Sentiment Analysis",
        "Aspect Analysis",
        "Complaint Analysis",
        "Product Comparison"
    ]
)


try:
    df = load_data()

    st.title("Customer Feedback Intelligence System")
    st.write("Analyze customer reviews, ratings and feedback.")

  
    if page == "Overview":

        st.header("Overview Dashboard")

        total_reviews = len(df)
        total_products = df["Product_name"].nunique()
        average_rating = df["Rate"].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )

        col2.metric(
            "Total Products",
            f"{total_products:,}"
        )

        col3.metric(
            "Average Rating",
            f"{average_rating:.2f} / 5"
            if pd.notna(average_rating)
            else "N/A"
        )

        st.subheader("Rating Distribution")

        rating_counts = (
            df["Rate"]
            .dropna()
            .loc[lambda x: x.isin([1, 2, 3, 4, 5])]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(10, 4.5))

        bars = ax.bar(
            rating_counts.index.astype(str),
            rating_counts.values
        )

        ax.set_xlabel("Rating")
        ax.set_ylabel("Reviews")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["1", "2", "3", "4", "5"])

        ax.bar_label(
            bars,
            labels=[
                f"{int(value):,}"
                for value in rating_counts.values
            ],
            padding=3,
            fontsize=9
        )

        ax.ticklabel_format(
            axis="y",
            style="plain"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Sample Reviews
        st.subheader("Sample Customer Reviews")

        st.dataframe(
            df[
                ["Product_name", "Rate", "Review", "Summary"]
            ].head(10),
            use_container_width=True
        )

   
    elif page == "Sentiment Analysis":

        st.header("Sentiment Analysis")

        sentiment_counts = (
            df["Sentiment"]
            .value_counts()
            .reindex(
                ["Positive", "Neutral", "Negative"],
                fill_value=0
            )
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Positive Reviews",
            f"{sentiment_counts['Positive']:,}"
        )

        col2.metric(
            "Neutral Reviews",
            f"{sentiment_counts['Neutral']:,}"
        )

        col3.metric(
            "Negative Reviews",
            f"{sentiment_counts['Negative']:,}"
        )

        st.subheader("Sentiment Distribution")

        chart_data = sentiment_counts

        fig, ax = plt.subplots(figsize=(10, 4.5))

        bars = ax.bar(
            chart_data.index,
            chart_data.values
        )

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Reviews")

        ax.bar_label(
            bars,
            labels=[
                f"{int(value):,}"
                for value in chart_data.values
            ],
            padding=3,
            fontsize=9
        )

        ax.ticklabel_format(
            axis="y",
            style="plain"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        
        st.subheader("Sentiment Percentage")

        total_reviews = len(df)

        if total_reviews > 0:
            sentiment_percent = (
                chart_data / total_reviews * 100
            ).round(2)

            percentage_df = (
                sentiment_percent
                .rename("Percentage (%)")
                .rename_axis("Sentiment")
                .reset_index()
            )

            st.dataframe(
                percentage_df,
                use_container_width=True
            )
        else:
            st.info("No customer reviews are available.")

    
   
    elif page == "Aspect Analysis":

        st.header("Aspect Analysis")

        try:
            aspect_df = pd.read_csv(
                ASPECT_PATH,
                encoding="utf-8"
            )

            required_columns = {"Aspect", "Sentiment"}

            if required_columns.issubset(aspect_df.columns):

                # 1. Graph first
                st.subheader("Sentiment by Aspect")

                aspect_counts = pd.crosstab(
                    aspect_df["Aspect"],
                    aspect_df["Sentiment"]
                )

                st.bar_chart(aspect_counts)

                # 2. Table after graph
                st.subheader("Aspect Sentiment Records")

                st.dataframe(
                    aspect_df.head(20),
                    use_container_width=True
                )

            else:
                st.warning(
                    "The CSV file must contain "
                    "'Aspect' and 'Sentiment' columns."
                )

        except FileNotFoundError:
            st.error(
                f"Aspect dataset not found: {ASPECT_PATH}"
            )

   
    elif page == "Complaint Analysis":

        st.header("Complaint Analysis")

        negative_df = df[
            df["Sentiment"] == "Negative"
        ].copy()

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Negative Reviews",
            f"{len(negative_df):,}"
        )

        col2.metric(
            "Products with Negative Reviews",
            f"{negative_df['Product_name'].nunique():,}"
        )

        # Negative Reviews by Rating
        st.subheader("Negative Reviews by Rating")

        negative_rating_counts = (
            negative_df["Rate"]
            .value_counts()
            .sort_index()
        )

        if not negative_rating_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))

            bars = ax.bar(
                negative_rating_counts.index.astype(int).astype(str),
                negative_rating_counts.values
            )

            ax.set_xlabel("Rating")
            ax.set_ylabel("Negative Reviews")

            ax.bar_label(
                bars,
                labels=[
                    f"{int(value):,}"
                    for value in negative_rating_counts.values
                ],
                padding=3,
                fontsize=9
            )

            ax.ticklabel_format(
                axis="y",
                style="plain"
            )

            ax.grid(
                axis="y",
                alpha=0.25
            )

            ax.set_axisbelow(True)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No negative reviews are available.")

        
        st.subheader("Sample Customer Complaints")

        st.dataframe(
            negative_df[
                ["Product_name", "Rate", "Review", "Summary"]
            ].head(20),
            use_container_width=True
        )

    
    
   
    
    elif page == "Product Comparison":

        st.header("Product Comparison")

        product_summary = (
            df.groupby("Product_name")
            .agg(
                Average_Rating=("Rate", "mean"),
                Total_Reviews=("Rate", "count")
            )
            .reset_index()
        )

        product_summary["Average_Rating"] = (
            product_summary["Average_Rating"].round(2)
        )

        # Select top 6 products
        top_products = (
            product_summary
            .nlargest(6, "Total_Reviews")
            .reset_index(drop=True)
        )

        product_names = top_products["Product_name"].tolist()

        st.subheader("Select a Product")

        # Six clickable product boxes
        for start in range(0, len(product_names), 3):
            cols = st.columns(3)

            for j, col in enumerate(cols):
                i = start + j

                if i < len(product_names):
                    with col:
                        if st.button(
                            product_names[i],
                            key=f"product_button_{i}",
                            use_container_width=True
                        ):
                            st.session_state["selected_product"] = (
                                product_names[i]
                            )

       
        if (
            "selected_product" not in st.session_state
            or st.session_state["selected_product"] not in product_names
        ):
            st.session_state["selected_product"] = product_names[0]

        selected_product = st.session_state["selected_product"]

       
        st.subheader("Top 6 Products by Review Count")

       
        graph_data = top_products.copy()

        graph_data["Product_Label"] = (
            graph_data["Product_name"]
            .astype(str)
            .apply(
                lambda name: name[:45] + "..."
                if len(name) > 45 else name
            )
        )

        
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(
            graph_data["Product_Label"],
            graph_data["Total_Reviews"]
        )

       
        for i, bar in enumerate(bars):
            if (
                graph_data.iloc[i]["Product_name"]
                == selected_product
            ):
                bar.set_color("orange")

        ax.set_xlabel("Reviews")
        ax.set_ylabel("Product Name")
        ax.invert_yaxis()

        ax.bar_label(
            bars,
            labels=[
                f"{int(value):,}"
                for value in graph_data["Total_Reviews"]
            ],
            padding=4,
            fontsize=9
        )

        ax.ticklabel_format(
            axis="x",
            style="plain"
        )

        ax.grid(
            axis="x",
            alpha=0.25
        )

        ax.set_axisbelow(True)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        
        selected_info = top_products[
            top_products["Product_name"] == selected_product
        ].iloc[0]

        st.divider()
        st.subheader("Selected Product Details")

        st.write(f"**Product Name:** {selected_product}")

        col1, col2 = st.columns(2)

        col1.metric(
            "Average Rating",
            f"{selected_info['Average_Rating']:.2f} / 5"
        )

        col2.metric(
            "Total Reviews",
            f"{int(selected_info['Total_Reviews']):,}"
        )

        
        st.subheader("Customer Reviews")

        selected_reviews = df[
            df["Product_name"] == selected_product
        ][["Product_name", "Rate", "Review", "Summary"]]

        st.dataframe(
            selected_reviews.head(20),
            use_container_width=True
        )


except FileNotFoundError:
    st.error(
        f"Dataset not found: {DATA_PATH}"
    )

    st.info(
        "Please check that the CSV file exists at the specified location."
    )

except KeyError as e:
    st.error(
        f"Required column not found in the dataset: {e}"
    )

except Exception as e:
    st.error(
        f"Unable to load the dashboard data: {e}"
    )
