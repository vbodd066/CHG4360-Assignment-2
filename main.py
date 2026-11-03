# =====================================================================
# MAIN PROGRAM
# Predicts the specific growth rate of a cell culture from temperature, pH, glucose and lactate.
# All of the work lives in src/, this file runs the workflow in order and prints a summary at the end.
# =====================================================================
import src.functions as sf


# 1. load the data and hold out a test set
df = sf.load_data()
x_train, x_test, y_train, y_test = sf.split_data(df)

# 2. tune each model family with cross-validation on the training set only
poly_search = sf.tune_model(*sf.build_polynomial_pipeline(), x_train, y_train)
knn_search = sf.tune_model(*sf.build_knn_pipeline(), x_train, y_train)

poly_table = sf.cv_table(poly_search)
knn_table = sf.cv_table(knn_search)
sf.save_table(poly_table.round(4), "polynomial_cv_results.csv")
sf.save_table(knn_table.round(4), "knn_cv_results.csv")
sf.plot_polynomial_validation(poly_table)
sf.plot_knn_validation(knn_table, best_k=knn_search.best_params_["kneighborsregressor__n_neighbors"])

# 3. evaluate both tuned models once on the test set
searches = {"Polynomial regression": poly_search, "K-nearest neighbours": knn_search}
evaluation = sf.evaluate_on_test(searches, x_train, x_test, y_train, y_test)

# 4. check whether more training data would improve either model
lc_table = sf.learning_curve(x_train, y_train, x_test, y_test)
sf.plot_learning_curve(lc_table)

# 5. check how each culture variable affects predicted growth
poly_model, knn_model = poly_search.best_estimator_, knn_search.best_estimator_
x_all, y_all = df[sf.FEATURES], df[sf.TARGET]
effects = sf.feature_effects(poly_model, knn_model, x_all)
sf.plot_feature_responses(poly_model, knn_model, x_all, y_all)

# 6. summary of the run
print("Test-set performance of the tuned models:")
print(evaluation[["model", "best_params", "test_r2", "test_rmse"]].round(3).to_string(index=False))
print("\nTables written to tables/ and figures written to figures/")
