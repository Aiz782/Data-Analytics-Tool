import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Load environment variables (if needed)
load_dotenv()

# Streamlit UI
st.title("Comprehensive Data Analysis App")
st.sidebar.title("Options")
menu = st.sidebar.selectbox("Choose an Option", ["Data Overview", "Univariate Analysis", "Bivariate Analysis"])

if menu == "Data Overview":
    st.subheader("Overview of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        st.write("**Preview of Uploaded Data:**")
        st.dataframe(df.head())
        
        # Basic information
        st.write("**Shape of the data:**", df.shape)
        st.write("**Data Information:**")
        st.write(df.info())
        st.write("**Missing values in each column:**")
        st.write(df.isnull().sum())
        st.write("**Descriptive statistics:**")
        st.write(df.describe(include='all').T)
        st.write("**Number of duplicated rows:**", df.duplicated().sum())
        
        # Select only numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.write("**Correlation Matrix:**")
            corr_matrix = numeric_df.corr()
            st.write(corr_matrix)
        else:
            st.write("No numeric data available for correlation matrix.")
        
        # Pandas Profiling Report
        st.write("**Pandas Profiling Report:**")
        profile = ProfileReport(df, title="Pandas Profiling Report")
        st_profile_report(profile)

elif menu == "Univariate Analysis":
    st.subheader("Univariate Analysis")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        st.write("**Preview of Uploaded Data:**")
        st.dataframe(df.head())
        
        # Selecting columns
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select a column for univariate analysis", columns)
        
        if selected_column:
            if df[selected_column].dtype == 'object':
                st.write(f"**Column:** {selected_column} is categorical.")
                st.write(df[selected_column].value_counts())
                st.write("**Count Plot:**")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=selected_column, ax=ax)
                st.pyplot(fig)
                
                st.write("**Pie Chart:**")
                fig, ax = plt.subplots()
                df[selected_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)
                
            else:
                st.write(f"**Column:** {selected_column} is numerical.")
                st.write(df[selected_column].describe())
                
                st.write("**Histogram:**")
                fig, ax = plt.subplots()
                sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
                st.pyplot(fig)
                
                st.write("**Dist Plot:**")
                fig, ax = plt.subplots()
                sns.distplot(df[selected_column], kde=True, ax=ax)
                st.pyplot(fig)
                
                st.write("**Box Plot:**")
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=selected_column, ax=ax)
                st.pyplot(fig)

elif menu == "Bivariate Analysis":
    st.subheader("Bivariate Analysis")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    
    if file_uploader is not None:
        df = pd.read_csv(file_uploader)
        st.write("**Preview of Uploaded Data:**")
        st.dataframe(df.head())
        
        # Selecting columns for bivariate analysis
        columns = df.columns.tolist()
        col1 = st.selectbox("Select the first column", columns)
        col2 = st.selectbox("Select the second column", columns)
        
        if col1 and col2:
            if df[col1].dtype != 'object' and df[col2].dtype != 'object':
                st.write(f"**Numerical vs Numerical Analysis**")
                st.write("**Scatter Plot:**")
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
                st.pyplot(fig)
                
            elif df[col1].dtype == 'object' and df[col2].dtype == 'object':
                st.write(f"**Categorical vs Categorical Analysis**")
                st.write("**Heatmap of Counts:**")
                counts = pd.crosstab(df[col1], df[col2])
                fig, ax = plt.subplots()
                sns.heatmap(counts, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
                
            else:
                st.write(f"**Categorical vs Numerical Analysis**")
                if df[col1].dtype == 'object':
                    cat_col, num_col = col1, col2
                else:
                    cat_col, num_col = col2, col1
                
                st.write("**Box Plot:**")
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)
                
                st.write("**Bar Plot:**")
                fig, ax = plt.subplots()
                sns.barplot(data=df, x=cat_col, y=num_col, ax=ax)
                st.pyplot(fig)

                st.write("**Dist Plot:**")
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=num_col, hue=cat_col, multiple="stack", ax=ax)
                st.pyplot(fig)
