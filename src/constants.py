# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

# column names in the raw dataset
TARGET = "Specific Growth Rate [d^-1]"
FEATURES = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]

# reproducibility and data splitting
RANDOM_STATE = 42
TEST_SIZE = 0.2                 # 20 % of samples are held out until the very end
CV_SPLITS = 4                   # folds used for cross-validation on the training set

# hyperparameter search spaces
POLY_DEGREES = [1, 2, 3, 4, 5]
KNN_NEIGHBORS = list(range(1, 41))
KNN_WEIGHTS = ["uniform", "distance"]

# training-set sizes for the learning curve (the full training set is always added as the last point)
LEARNING_CURVE_SIZES = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

# reference culture condition used when sweeping one variable at a time (close to the dataset medians)
REFERENCE_CONDITION = {"Temperature [C]": 37.0, "pH": 7.0, "Glucose [mM]": 15.0, "Lactate [mM]": 15.0}

# number of points in each one-variable sweep
SWEEP_POINTS = 200
