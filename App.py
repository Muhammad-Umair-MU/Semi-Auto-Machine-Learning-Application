import streamlit as st 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Models
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC



def main():

	st.title("Semi-Auto Machine Learning Application")

	activities = ["Exploratory Data Analysis","Plots","Model Building"]
	choice = st.sidebar.selectbox("Select Activity ",activities)

	if choice == "Exploratory Data Analysis":
		st.subheader("Exploratory Data Analysis")
		data = st.file_uploader("Upload Dataset",type=["CSV"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape"):
				st.write(df.shape)
			if st.checkbox("Show Columns"):
				all_columns = df.columns.tolist()
				st.write(all_columns)
			if st.checkbox("Selected Columns to Show"):
				all_columns = df.columns.tolist()
				selected_colums = st.multiselect("Select Columns",all_columns)
				new_data_frame = df[selected_colums]
				st.dataframe(new_data_frame)
			if st.checkbox("Show Summary"):
				st.write(df.describe())
			if st.checkbox("Show Value Columns"):
				st.write(df.iloc[:,-1].value_counts())
	elif  choice == "Plots":
		st.subheader("Data Visualizations")
		data = st.file_uploader("Upload Dataset",type=["CSV"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
		if st.checkbox("Correlation with Seaborn"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()
		if st.checkbox("Correlation with Pie Chart"):
			all_columns = df.columns.tolist()
			columns_to_plot = st.selectbox("Select Column",all_columns)
			pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="1%.1f%%")
			st.write(pie_plot)
			st.pyplot()

		all_columns_name = df.columns.tolist()
		plots = st.selectbox("Select Type of Plot",["Area","Bar","Hist","Box"])
		selected_colums_names = st.multiselect("Select Columns to Plot",all_columns_name)
		if st.button("Generate Plots"):
			st.success("Generating Customizable Plot of {} for {}".format(plots,selected_colums_names))

			if plots == "Area":
				custom_data = df[selected_colums_names]
				st.area_chart(custom_data)
			elif plots == "Bar":
				custom_data = df[selected_colums_names]
				st.bar_chart(custom_data)
			elif plots == "Hist":
				custom_data = df[selected_colums_names]
				st.Hist(custom_data)
			elif plots:
				custom_plot = df[selected_colums_names].plot(kind=plots)
				st.write(custom_plot)
				st.pyplot()
				


	elif  choice == "Model Building":
		st.subheader("Model Building")
		data = st.file_uploader("Upload Dataset",type=["CSV"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
		# Model Building
			X = df.iloc[:,0:-1]
			Y = df.iloc[:,-1]
			seed = 7

			# Model
			models = []
			models.append(("LR", LogisticRegression()))
			models.append(("LDA", LinearDiscriminantAnalysis()))
			models.append(("KNN", KNeighborsClassifier()))
			models.append(("Cart", DecisionTreeClassifier()))
			models.append(("NB", GaussianNB()))
			models.append(("SVM", SVC()))
			# Evaluations
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = "accuracy"

			for names,model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
				model_names.append(names)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())

				accuracy_result = {"Model Name":names,"Model Accuracy":cv_results.mean(),"Standard Deviation":cv_results.std()}
				all_models.append(accuracy_result)

			if st.checkbox("Metrics as Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))
			if st.checkbox("Metrics as JSON"):
				st.json(all_models)

	elif  choice == "Model Building":
		st.subheader("Model Building")
main()