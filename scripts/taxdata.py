"""Taxonomy data + the shared page header / family column used by every category page."""
from taxdata_raw import CATS as _RAW
from dio import Page, E, HL, BLACK, GREY

PARADIGMS = ["Supervised Learning", "Unsupervised Learning", "Deep Learning & NLP",
             "Reinforcement Learning", "Time Series", "Foundations"]

# background concepts each base algorithm is built on (shown as the 'Built on' chain)
BUILT_ON = {
    "regression": ["Mean / Variance", "Least Squares", "Cost Function (MSE)", "Gradient Descent"],
    "classification": ["Linear Regression", "Sigmoid Function", "Log Loss (Cross-Entropy)", "Gradient Descent"],
    "kernel": ["Feature Scaling", "Distance Metrics", "Majority Vote / Average", "Cross-Validation (choose k)"],
    "tree": ["Probability", "Entropy", "Information Gain", "Recursive Partitioning"],
    "ensemble": ["Bootstrap Sampling", "Decision Tree (base learner)", "Majority Vote / Averaging", "Bias–Variance Trade-off"],
    "clustering": ["Euclidean Distance", "Centroid (Mean)", "Iterative Assign–Update", "Elbow / Silhouette"],
    "dimred": ["Standardization", "Covariance Matrix", "Eigen-decomposition", "Explained Variance"],
    "anomaly": ["Mean (μ)", "Standard Deviation (σ)", "Normal Distribution", "68–95–99.7 Rule"],
    "association": ["Set Theory / Itemsets", "Support", "Confidence & Lift", "Apriori Property"],
    "deeplearning": ["Perceptron (Neuron)", "Activation Functions", "Forward Propagation", "Backpropagation + Gradient Descent"],
    "nlp": ["Text Cleaning", "Tokenization", "Vocabulary", "Vector Space (Document-Term Matrix)"],
    "rl": ["Markov Decision Process", "Reward & Discount γ", "Bellman Equation", "ε-Greedy Exploration"],
    "timeseries": ["Stationarity", "Autocorrelation (ACF / PACF)", "White Noise", "Least Squares / MLE"],
    "optimization": ["Derivative / Slope", "Partial Derivatives (Gradient)", "Cost Function", "Learning Rate α"],
    "distributions": ["Random Variable", "Probability (PMF / PDF)", "Central Tendency", "Dispersion & Shape"],
}

DISTRIBUTIONS = dict(
    id="distributions", name="Distributions & Descriptive Statistics", par="Foundations",
    desc="How data is spread: distribution shapes and the statistics that describe them.",
    groups=[
        ("Continuous Distributions", [("Normal (Gaussian)", "Symmetric bell curve, μ and σ"),
                                      ("Uniform", "Every value equally likely"),
                                      ("Exponential", "Time between events"),
                                      ("Log-Normal", "log(x) is normal, right-skewed"),
                                      ("Student's t", "Heavier tails, small samples"),
                                      ("Chi-Square", "Sum of squared normals")]),
        ("Discrete Distributions", [("Bernoulli", "One trial: success / failure"),
                                    ("Binomial", "Successes in n trials"),
                                    ("Poisson", "Events per interval"),
                                    ("Geometric", "Trials until first success")]),
        ("Shape", [("Symmetric", "Mean = Median = Mode"),
                   ("Right (Positive) Skew", "Mode < Median < Mean"),
                   ("Left (Negative) Skew", "Mean < Median < Mode"),
                   ("Bimodal / Multimodal", "Two or more peaks"),
                   ("Kurtosis", "Lepto / Meso / Platykurtic")]),
        ("Descriptive Statistics", [("Mean / Median / Mode", "Central tendency"),
                                    ("Variance / Std. Dev. (σ)", "Dispersion"),
                                    ("Range / Quartiles / IQR", "Spread & box plot"),
                                    ("Skewness / Kurtosis", "Shape measures")]),
    ],
    pick="Normal (Gaussian)")

STATISTICS = dict(
    id="statistics", name="Inferential Statistics", par="Foundations",
    desc="Use a sample to draw conclusions about a population: sampling, CLT, estimation and hypothesis testing.",
    groups=[
        ("Sampling Theory", [("Population vs Sample", "Parameters (μ, σ) vs statistics (x̄, s)"),
                             ("Law of Large Numbers", "x̄ → μ as n grows"),
                             ("Central Limit Theorem (CLT)", "Sample means become normal"),
                             ("Standard Error", "SE = σ / √n")]),
        ("Estimation", [("Point Estimate", "x̄, s, p̂"),
                        ("Confidence Interval", "x̄ ± z·SE"),
                        ("Maximum Likelihood (MLE)", "Parameters that best explain data"),
                        ("Bootstrap", "Resampling to estimate uncertainty")]),
        ("Hypothesis Testing", [("Z-Test", "σ known or large n"),
                                ("t-Test (1-sample / 2-sample / paired)", "σ unknown, small n"),
                                ("Chi-Square Test", "Categorical independence / fit"),
                                ("ANOVA (F-test)", "Compare 3+ group means")]),
        ("Decision & Errors", [("p-value & Significance α", "Evidence against H₀"),
                               ("Type I / Type II Error", "False positive / false negative"),
                               ("Statistical Power", "1 − β"),
                               ("Correlation Tests", "Pearson / Spearman")]),
    ],
    pick="Central Limit Theorem (CLT)")

BUILT_ON["statistics"] = ["Random Sampling", "Mean & Variance", "Normal Distribution", "Law of Large Numbers"]

DATANATURE = dict(
    id="datanature", name="Nature of Data (Tabular)", par="Foundations",
    desc="Know your columns before modelling: data types, levels of measurement, relationship shapes and structure.",
    groups=[
        ("Qualitative (Categorical)", [("Nominal", "Labels, no order: sector, colour"),
                                       ("Ordinal", "Ordered labels: low / mid / high, ratings"),
                                       ("Binary", "Two values: yes / no, up / down")]),
        ("Quantitative (Numerical)", [("Discrete", "Countable: number of trades"),
                                      ("Continuous", "Measurable: price, return, time")]),
        ("Measurement Scales", [("Levels of Measurement (NOIR)", "Nominal → Ordinal → Interval → Ratio"),
                                ("Interval", "Equal steps, no true zero: °C, dates"),
                                ("Ratio", "True zero: price, volume, age")]),
        ("Relationship Shape", [("Linear", "Constant rate: Pearson r"),
                                ("Monotonic (non-linear)", "Always ↑ or always ↓: Spearman ρ / Kendall τ"),
                                ("Non-monotonic", "U-shape, cyclic: mutual information"),
                                ("No relationship", "Independent variables")]),
        ("Data Structure & Quality", [("Cross-sectional / Time Series / Panel", "One time, one entity, or both"),
                                      ("Missing Data (MCAR / MAR / MNAR)", "Why values are missing"),
                                      ("Imbalanced / Outliers", "Rare classes, extreme values")]),
    ],
    pick="Levels of Measurement (NOIR)")

HYPOTESTS = dict(
    id="hypotests", name="Hypothesis Tests by Data Type", par="Foundations",
    desc="Choose the right test from the types of variables involved and how the samples are linked.",
    groups=[
        ("Assumption Checks", [("Shapiro-Wilk", "Normality (small/medium n)"),
                               ("Kolmogorov-Smirnov", "Sample vs reference distribution"),
                               ("Levene / Bartlett", "Equal variances"),
                               ("Q-Q Plot", "Visual normality check")]),
        ("Numeric ~ Groups (parametric)", [("Z-Test", "Mean vs μ₀, σ known / large n"),
                                           ("One-sample t-Test", "Mean vs μ₀, σ unknown"),
                                           ("Independent t-Test", "2 independent groups"),
                                           ("Paired t-Test", "2 linked (before/after) measurements"),
                                           ("One-way ANOVA", "3+ independent groups"),
                                           ("Repeated-measures ANOVA", "3+ linked measurements")]),
        ("Numeric ~ Groups (non-parametric)", [("Wilcoxon Signed-Rank", "Paired / one-sample, medians"),
                                               ("Mann-Whitney U", "2 independent groups, ranks"),
                                               ("Kruskal-Wallis", "3+ independent groups, ranks"),
                                               ("Friedman", "3+ linked measurements, ranks")]),
        ("Categorical ~ Categorical", [("Chi-Square Goodness-of-Fit", "Observed vs expected counts"),
                                       ("Chi-Square Independence", "Contingency table"),
                                       ("Fisher's Exact", "Small expected counts (< 5)"),
                                       ("McNemar", "Paired binary outcomes")]),
        ("Numeric ~ Numeric", [("Pearson Correlation", "Linear association"),
                               ("Spearman Rank", "Monotonic association"),
                               ("Kendall Tau", "Ordinal / small samples"),
                               ("Regression t / F-test", "Slope ≠ 0, model significance")]),
    ],
    pick="Z-Test")

TSTESTS = dict(
    id="tstests", name="Time Series Hypothesis Tests & Stationarity", par="Time Series",
    desc="Test whether a series is stationary, autocorrelated, related to another series, and how to fix it.",
    groups=[
        ("Stationarity / Unit-Root Tests", [("Augmented Dickey-Fuller (ADF)", "H₀: unit root (non-stationary)"),
                                            ("KPSS", "H₀: stationary (opposite of ADF)"),
                                            ("Phillips-Perron (PP)", "Unit root, robust to heteroscedasticity"),
                                            ("Zivot-Andrews", "Unit root allowing one structural break")]),
        ("Types of Non-stationarity", [("Trend", "Mean drifts over time"),
                                       ("Seasonality", "Repeating calendar pattern"),
                                       ("Random Walk (unit root)", "Shocks never fade"),
                                       ("Structural Break", "Sudden regime change"),
                                       ("Changing Variance", "Volatility clustering")]),
        ("Residual / Dependence Tests", [("Ljung-Box", "H₀: no autocorrelation up to lag h"),
                                         ("Durbin-Watson", "Lag-1 autocorrelation of residuals"),
                                         ("ARCH-LM (Engle)", "H₀: no volatility clustering"),
                                         ("Breusch-Godfrey", "Higher-order serial correlation")]),
        ("Relationship Tests", [("Granger Causality", "Does X help predict Y?"),
                                ("Engle-Granger Cointegration", "Two non-stationary series, stationary spread"),
                                ("Johansen", "Cointegration among many series")]),
        ("Make it Stationary", [("Differencing", "yₜ − yₜ₋₁"),
                                ("Log / Box-Cox", "Stabilise variance"),
                                ("Detrending", "Remove fitted trend"),
                                ("Seasonal Differencing", "yₜ − yₜ₋ₛ"),
                                ("Fractional Differencing", "Stationary but keeps memory")]),
    ],
    pick="Augmented Dickey-Fuller (ADF)")

LABELLING = dict(
    id="labelling", name="Time Series Data Labelling", par="Time Series",
    desc="Turn a price series into supervised-learning labels without look-ahead bias (financial ML).",
    groups=[
        ("Sampling / Bars", [("Time Bars", "Every fixed time interval"),
                             ("Volume / Dollar Bars", "Every fixed traded amount"),
                             ("CUSUM Filter", "Sample only when a move accumulates")]),
        ("Labelling Methods", [("Fixed-Time Horizon", "Sign of return after h bars vs threshold τ"),
                               ("Triple Barrier Method", "Profit-take / stop-loss / time barrier"),
                               ("Meta-Labelling", "2nd model decides: act on the signal or not"),
                               ("Trend-Scanning", "Label by the most significant trend t-value")]),
        ("Sample Weights", [("Label Concurrency / Uniqueness", "Overlapping labels share information"),
                            ("Return Attribution", "Weight by absolute return"),
                            ("Time Decay", "Recent samples weigh more")]),
        ("Leakage-safe Validation", [("Purged K-Fold CV", "Drop train labels overlapping the test set"),
                                     ("Embargo", "Gap after the test fold"),
                                     ("Walk-Forward", "Train past → test future, rolling")]),
    ],
    pick="Fixed-Time Horizon")

BUILT_ON.update({
    "datanature": ["Variables & Observations", "Qualitative vs Quantitative", "Measurement Scale", "Relationship Shape"],
    "hypotests": ["Data Types (NOIR)", "Sampling Distribution (CLT)", "Test Statistic", "p-value & α"],
    "tstests": ["Stationarity", "Unit Root (Random Walk)", "Autoregression", "Hypothesis Test (p-value)"],
    "labelling": ["Returns & Volatility", "Event Sampling (CUSUM)", "Thresholds / Barriers", "Leakage-safe Validation"],
})

# assumptions of the base algorithm: (assumptions, how to check, if violated)
ASSUMPTIONS = {
    "regression": (["Linearity between x and y", "Independent errors", "Normal residuals ε ~ N(0, σ²)",
                    "Homoscedasticity (constant variance)", "No influential outliers",
                    "(Multiple LR) no multicollinearity"],
                   "Residual plot, Q-Q plot, Durbin-Watson, VIF, Cook's distance",
                   "Transform x / y, add polynomial terms, robust or regularized regression"),
    "classification": (["Binary (or categorical) target", "Independent observations",
                        "Linear relation between features and log-odds", "No strong multicollinearity",
                        "Enough samples (≈10+ events per feature)"],
                       "Box-Tidwell test, VIF, class counts",
                       "Add interactions / splines, regularize (L1/L2), use trees or kernels"),
    "kernel": (["Similar points have similar labels (smoothness)", "Features on comparable scales",
                "Only relevant features (all count equally in distance)", "Low–moderate dimensionality",
                "No distribution assumption (non-parametric)"],
               "Scale check, CV error vs k, feature relevance",
               "Standardize, feature selection / PCA, weighted or metric-learned kNN"),
    "tree": (["Categorical features (continuous must be binned)", "Categorical target",
              "No distribution assumption", "Training data representative of future data",
              "No missing values"],
             "Data types, missing-value report",
             "C4.5 / CART for numeric & missing values, pruning for overfitting"),
    "ensemble": (["Base learner is unstable (high variance, low bias)", "i.i.d. observations",
                  "Bootstrap samples represent the population", "Base models' errors not perfectly correlated"],
                 "Variance of single model across bootstraps, OOB error",
                 "Random Forest (decorrelate), boosting for high bias, purged bootstraps for time series"),
    "clustering": (["Spherical (isotropic) clusters", "Similar cluster size and density", "k known in advance",
                    "Numeric, scaled features (Euclidean distance)", "Few outliers"],
                   "Elbow / silhouette, scatter / PCA plot",
                   "GMM (elliptic), DBSCAN (any shape, noise), K-Medoids (outliers)"),
    "dimred": (["Linear relationships between features", "High variance = important information",
                "Standardized numeric features", "Sufficient correlation (KMO > 0.6, Bartlett's test)",
                "Few outliers"],
               "Correlation matrix, KMO, Bartlett, scree plot",
               "Kernel PCA / t-SNE / UMAP (non-linear), robust PCA (outliers)"),
    "anomaly": (["Data approximately normal", "One feature at a time (univariate)",
                 "μ and σ not distorted by the outliers themselves", "Independent observations",
                 "Stationary data (for time series)"],
                "Histogram, Q-Q plot, Shapiro-Wilk",
                "Modified Z (median / MAD), IQR rule, Isolation Forest (multivariate)"),
    "association": (["Transactional (binary presence) data", "Item order does not matter",
                     "Downward closure: subsets of frequent sets are frequent", "Enough transactions"],
                    "Support distribution, number of rules",
                    "FP-Growth for speed, sequential patterns (GSP) when order matters"),
    "deeplearning": (["i.i.d. samples", "Scaled numeric inputs", "Enough data relative to parameters",
                      "Differentiable loss and activations", "Fixed-size inputs, no spatial / temporal structure"],
                     "Learning curves (train vs validation), input statistics",
                     "CNN (images), RNN / LSTM (sequences), regularization / more data"),
    "nlp": (["Word order does not matter ('bag')", "Words are independent",
             "Vocabulary covers new text (no out-of-vocabulary words)", "Frequency ≈ importance (after stop-words)"],
            "OOV rate, vocabulary size",
            "N-grams (order), TF-IDF (importance), embeddings / BERT (meaning)"),
    "rl": (["Markov property: next state depends only on current state & action", "Stationary environment",
            "Finite, discrete states and actions (tabular)", "Every state–action pair visited often",
            "Suitable decaying learning rate"],
           "State-visit counts, reward curve stability",
           "DQN (large / continuous states), POMDP methods, add history to the state"),
    "timeseries": (["Weak stationarity (constant mean, variance, autocovariance)", "Linear dependence on the past",
                    "Residuals are white noise", "Causal |φ| < 1 and invertible MA", "Regular time intervals, no breaks"],
                   "ADF / KPSS, ACF of residuals, Ljung-Box",
                   "Difference (ARIMA), SARIMA (season), GARCH (variance), break dummies"),
    "optimization": (["Differentiable cost function", "Convex cost → global minimum guaranteed",
                      "Smooth gradient (Lipschitz) so a small α converges", "Scaled features",
                      "Data fits in memory (batch)"],
                     "Cost-vs-iteration curve, gradient norm",
                     "SGD / mini-batch (big data), momentum / Adam (ravines), restarts (non-convex)"),
    "distributions": (["Continuous variable", "Symmetric and unimodal", "i.i.d. observations",
                       "Thin tails (no extreme outliers)"],
                      "Histogram, Q-Q plot, Shapiro-Wilk, skewness / kurtosis",
                      "Transform (log / Box-Cox), use t / log-normal / non-parametric methods"),
    "statistics": (["Random sampling", "Independent observations (≤ 10 % of population)", "Finite variance σ²",
                    "Large enough n (≥ 30, more if skewed)", "Identically distributed"],
                   "Sampling design, skewness, n",
                   "Bootstrap, larger n, robust / non-parametric statistics"),
    "datanature": (["Arithmetic (mean, σ) only on interval / ratio data", "Order only meaningful from ordinal up",
                    "Ratios ('twice as much') only on ratio data", "Encoding must match the type"],
                   "Data dictionary, unique values per column",
                   "Median / mode for ordinal, one-hot for nominal, rank methods"),
    "hypotests": (["Random, independent samples", "Population σ known (or n ≥ 30)",
                   "Normal population or large n (CLT)", "Interval / ratio data"],
                  "Shapiro-Wilk, Q-Q plot, sample design",
                  "t-test (σ unknown), non-parametric tests (non-normal, ordinal)"),
    "tstests": (["Series follows an AR process (lags absorb autocorrelation)", "Correct lag length (AIC / BIC)",
                 "Correct deterministic terms (constant / trend)", "No structural break",
                 "Low power near a unit root"],
                "Residual ACF, plot of the series, KPSS as a cross-check",
                "Zivot-Andrews (breaks), PP (heteroscedasticity), confirm with KPSS"),
    "labelling": (["Volatility roughly constant (fixed threshold τ)", "Horizon h is meaningful for the strategy",
                   "No look-ahead when computing returns", "Labels independent (no overlap)"],
                  "Rolling volatility plot, label balance, label overlap",
                  "Triple Barrier with dynamic σ, sample-uniqueness weights, purged CV"),
}

EVALUATION = dict(
    id="evaluation", name="Model Evaluation & Validation", par="ML Workflow",
    desc="Measure how good a model really is on unseen data, and diagnose why it fails.",
    groups=[
        ("Validation Strategy", [("Hold-out (Train / Validation / Test)", "Fit, tune and report on separate data"),
                                 ("K-Fold Cross-Validation", "Every row is tested once"),
                                 ("Stratified K-Fold", "Keeps class ratios per fold"),
                                 ("Time-Series Split", "Train on the past, test on the future"),
                                 ("Purged K-Fold + Embargo", "No label overlap (finance)")]),
        ("Regression Metrics", [("MAE", "Mean absolute error"),
                                ("MSE / RMSE", "Penalises large errors"),
                                ("R²", "Variance explained"),
                                ("MAPE", "Percentage error")]),
        ("Classification Metrics", [("Confusion Matrix", "TP, FP, FN, TN"),
                                    ("Accuracy / Precision / Recall / F1", "Threshold-based scores"),
                                    ("ROC-AUC / PR-AUC", "Threshold-free ranking quality"),
                                    ("Log Loss / Brier", "Probability quality, calibration")]),
        ("Diagnostics", [("Bias–Variance Trade-off", "Under- vs over-fitting"),
                         ("Learning Curves", "Error vs training size"),
                         ("Validation Curves", "Error vs hyperparameter"),
                         ("Data Leakage Checks", "Future / target information in features")]),
    ],
    pick="Hold-out (Train / Validation / Test)")

FEATUREENG = dict(
    id="featureeng", name="Feature Engineering", par="ML Workflow",
    desc="Turn raw columns into informative, model-ready features.",
    groups=[
        ("Scaling & Transformation", [("Standardization (Z-score)", "(x − μ) / σ"),
                                      ("Min-Max Normalization", "Rescale to [0, 1]"),
                                      ("Robust Scaling", "(x − median) / IQR"),
                                      ("Log / Box-Cox / Yeo-Johnson", "Reduce skew")]),
        ("Encoding (categorical)", [("One-Hot Encoding", "Nominal → 0/1 columns"),
                                    ("Ordinal Encoding", "Ordered labels → 1, 2, 3"),
                                    ("Target / Mean Encoding", "Category → mean target (with CV)"),
                                    ("Binary / Hashing", "High-cardinality categories")]),
        ("Discretization", [("Equal-width Binning", "Same interval size"),
                            ("Equal-frequency Binning", "Same count per bin"),
                            ("Binarization", "Threshold → 0/1")]),
        ("Feature Creation", [("Interaction / Polynomial", "x₁·x₂, x²"),
                              ("Lag & Rolling Features", "yₜ₋ₖ, rolling mean / std"),
                              ("Technical Indicators", "EMA, RSI, MACD, ATR, Bollinger"),
                              ("Date / Time Features", "Hour, weekday, cyclical sin/cos")]),
        ("Missing Values & Selection", [("Imputation", "Mean / median / model-based"),
                                        ("Filter / Wrapper / Embedded Selection", "Correlation, RFE, Lasso")]),
    ],
    pick="Standardization (Z-score)")

EXPLAIN = dict(
    id="explain", name="Explainability (XAI)", par="ML Workflow",
    desc="Understand which features drive a model's predictions, globally and for one prediction.",
    groups=[
        ("Global (whole model)", [("Model Coefficients", "Linear / logistic weights"),
                                  ("Impurity (Gini) Importance", "Tree split gains"),
                                  ("Permutation Importance", "Score drop when a feature is shuffled"),
                                  ("Partial Dependence (PDP)", "Average effect of one feature")]),
        ("Local (one prediction)", [("SHAP", "Shapley values: fair share of each feature"),
                                    ("LIME", "Local linear surrogate"),
                                    ("ICE Curves", "PDP per individual row"),
                                    ("Counterfactuals", "Smallest change that flips the output")]),
        ("Model-specific", [("Tree SHAP", "Exact & fast for tree ensembles"),
                            ("Attention Weights", "Transformers (with caution)"),
                            ("Saliency / Grad-CAM", "CNN pixel importance")]),
    ],
    pick="Permutation Importance")

BUILT_ON.update({
    "evaluation": ["Generalization", "Train / Test Separation", "Error Metrics", "Bias–Variance"],
    "featureeng": ["Data Types (NOIR)", "Distributions (μ, σ, skew)", "Domain Knowledge", "No Leakage (fit on train)"],
    "explain": ["Trained Model", "Validation Metric", "Feature Perturbation", "Shapley Values (game theory)"],
})

ASSUMPTIONS.update({
    "evaluation": (["Test data comes from the same distribution as future data", "Test set is never used for training or tuning",
                    "Rows are independent (else use grouped / time-series splits)", "Metric matches the business goal"],
                   "Compare train / validation / test scores, check splits by time and group",
                   "Time-series or purged CV, nested CV for tuning, choose cost-based metrics"),
    "featureeng": (["Scaler / encoder statistics are fit on training data only", "Feature roughly symmetric for z-score to be meaningful",
                    "Same transformation applied at prediction time", "No future information in engineered features"],
                   "Pipeline review, distribution plots before / after, leakage tests",
                   "Use sklearn Pipelines, robust scaling for outliers, shift lagged features"),
    "explain": (["Model is reasonably accurate (explaining a bad model misleads)", "Features are not strongly correlated",
                 "Importance measured on held-out data", "Shuffling a feature creates realistic rows"],
                "Correlation matrix, compare importances on train vs test, repeat with several seeds",
                "Group correlated features, conditional permutation, SHAP with interventional background"),
})

FITTING = dict(
    id="fitting", name="Overfitting & Underfitting", par="ML Workflow",
    desc="How the shape and quality of a data table push a model into under- or over-fitting, and how to fix it.",
    groups=[
        ("Diagnosis", [("Bias–Variance Trade-off", "Error = bias² + variance + noise"),
                       ("Train vs Validation Gap", "Small gap & high error = underfit; big gap = overfit"),
                       ("Learning / Validation Curves", "Error vs data size / complexity")]),
        ("Underfitting Causes", [("Model too simple", "Straight line on a curved pattern"),
                                 ("Missing / weak features", "Signal not in the table"),
                                 ("Too much regularization", "λ too large"),
                                 ("Too little training", "Stopped too early")]),
        ("Overfitting Causes", [("Too many columns for the rows (p ≫ n)", "Model memorises rows"),
                                ("ID / leaky columns", "Unique or future information"),
                                ("Noisy labels & outliers", "Model fits the noise"),
                                ("Model too complex", "Deep trees, high-degree polynomials")]),
        ("Remedies", [("Regularization (L1 / L2)", "Penalise large weights"),
                      ("Pruning / max_depth / min_samples", "Limit tree growth"),
                      ("Early stopping & Dropout", "Neural networks"),
                      ("More data / Augmentation", "Reduce variance"),
                      ("Feature selection / Drop leaky columns", "Remove spurious signal"),
                      ("Cross-validation & Ensembling", "Honest scores, averaged models")]),
    ],
    pick="Bias–Variance Trade-off")

VERSATILE = dict(
    id="versatile", name="Versatile Uses (Hidden Tricks)", par="ML Workflow",
    desc="Familiar algorithms used for unexpected jobs: imputation, image editing, compression, features, signals.",
    groups=[
        ("Nearest-Neighbour Tricks", [("KNN Pixel Inpainting", "Fill missing pixels from the closest colours"),
                                      ("KNN Imputer", "Fill missing table values from similar rows"),
                                      ("kNN Distance Anomaly Score", "Far from neighbours = odd"),
                                      ("Similarity Recommendation", "Users / assets like this one")]),
        ("Clustering Tricks", [("K-Means Colour Quantization", "Image with only k colours"),
                               ("Cluster-ID / Distance Features", "New features for a supervised model"),
                               ("Market Regime Labels", "Cluster returns & volatility")]),
        ("Projection Tricks", [("PCA Denoising", "Reconstruct from top components"),
                               ("PCA Compression / Eigen-images", "Store few components"),
                               ("Whitening / Decorrelation", "Remove multicollinearity")]),
        ("Model-as-a-Tool", [("Tree Surrogate", "Explain a black box with rules"),
                             ("Forest Proximity", "Impute & find outliers"),
                             ("Autoencoder Anomaly", "Large reconstruction error"),
                             ("Platt Scaling (Logistic)", "Calibrate any model's scores")]),
        ("Statistical Trading Tricks", [("Regression Hedge Ratio", "β of pair → spread"),
                                        ("Z-score Mean Reversion", "Enter at |z| > 2"),
                                        ("ARMA Residual Alerts", "Unexpected moves"),
                                        ("Apriori Signal Mining", "Indicators that fire together")]),
    ],
    pick="KNN Pixel Inpainting")

BUILT_ON.update({
    "fitting": ["Train / Validation Split", "Model Complexity", "Data Size vs Features", "Regularization"],
    "versatile": ["Distance & Similarity", "Clustering", "Projection", "Reconstruction Error"],
})
ASSUMPTIONS.update({
    "fitting": (["Validation data is independent of training data", "Same distribution in train and future data",
                 "Metric reflects the real objective", "Noise level is roughly stable"],
                "Train vs validation curves, repeated CV, check for leakage",
                "Time-ordered / grouped splits, robust metrics, collect cleaner labels"),
    "versatile": (["Nearby points (pixels / rows) are similar (smoothness)", "Distance is meaningful (scaled features)",
                   "Enough complete neighbours around a gap", "The trick is validated like any model"],
                  "Hide known values, re-fill them and measure the error",
                  "Use model-based imputation, larger k, or domain-specific methods"),
})

# '💡 Also used for' box on every page
HIDDEN_USES = {
    "regression": ["Hedge ratio β for pairs trading", "Detrending a series (fit & subtract)", "Trend-slope feature over a window",
                   "Normal-equation core of many models"],
    "classification": ["Platt scaling: calibrate any model's scores", "Meta-labelling: bet size from P(success)",
                       "Propensity scores (A/B, causal inference)", "Baseline for every classifier"],
    "kernel": ["Pixel inpainting from closest colours", "KNN imputer for missing table values",
               "Anomaly score = distance to k-th neighbour", "Recommendations: 'similar users / assets'"],
    "tree": ["Rule extraction (IF-THEN trading rules)", "Supervised binning of a feature",
             "Surrogate model to explain a black box", "Quick feature-importance screen"],
    "ensemble": ["OOB error = free validation", "Prediction spread = uncertainty",
                 "Forest proximity → imputation & outliers", "Feature selection by importance"],
    "clustering": ["Colour quantization / image compression", "Cluster-ID & distance as new features",
                   "Market regime labelling", "Initialising GMM or RBF networks"],
    "dimred": ["Denoising by reconstruction", "Compression (eigen-images)", "Decorrelate features before regression",
               "Factor models of asset returns"],
    "anomaly": ["Feature standardization", "Mean-reversion entry signal (|z| > 2)", "Data-quality checks in pipelines",
                "Control charts (manufacturing)"],
    "association": ["Mining indicators that fire together", "Log / error pattern mining", "Feature co-occurrence discovery",
                    "Cross-selling bundles"],
    "deeplearning": ["Autoencoder anomaly detection", "Learned embeddings for categories", "Universal function approximator for any regression",
                     "Surrogate of slow simulations"],
    "nlp": ["Document search & similarity", "Spam / topic baseline with Naive Bayes", "Encoding log messages for anomaly detection",
            "Counting tokens of any sequence (e.g. candle patterns)"],
    "rl": ["Order-execution optimisation", "Hyperparameter / architecture search", "Dynamic position sizing",
           "Recommendation as sequential decisions"],
    "timeseries": ["Residual-based anomaly alerts", "Baseline forecaster for any model", "Generating synthetic series",
                   "Pre-whitening before cross-correlation"],
    "optimization": ["Fitting any differentiable model", "Portfolio weight optimisation", "Image style transfer (optimise pixels)",
                     "Adversarial examples"],
    "distributions": ["Value-at-Risk (normal / t quantiles)", "Simulation (Monte Carlo)", "Priors in Bayesian models",
                      "Synthetic data generation"],
    "statistics": ["Bootstrap confidence intervals for any metric", "A/B testing strategies", "Monte Carlo error bars",
                   "Sample-size planning"],
    "datanature": ["Choosing encoders automatically from column types", "Data-validation rules", "Picking the right chart",
                   "Selecting correlation measures"],
    "hypotests": ["Comparing two models' CV scores (paired t)", "Feature selection (χ², ANOVA F)", "Drift detection (KS test)",
                  "Strategy vs benchmark returns"],
    "tstests": ["Pairs selection by cointegration", "Choosing differencing order d", "Regime-break detection",
                "Validating residuals of any forecaster"],
    "labelling": ["Event-driven backtests", "Sample weights for any classifier", "Bet sizing via meta-labels",
                  "Leakage-free CV for any time series"],
    "evaluation": ["Threshold tuning for trading costs", "Model monitoring in production", "Drift alarms (metric decay)",
                   "Comparing feature sets"],
    "featureeng": ["Scaling for faster gradient descent", "Cyclical encoding for any periodic value", "Target encoding for high-cardinality IDs",
                   "Lag features turn any regressor into a forecaster"],
    "explain": ["Feature selection (drop low importance)", "Debugging leakage (suspicious top feature)", "Regulatory reporting",
                "Building trust with stakeholders"],
    "fitting": ["Choosing model size for a data budget", "Deciding whether to collect more data", "Setting early-stopping patience",
                "Detecting leakage (too-good validation)"],
    "versatile": ["Image inpainting & restoration", "Missing-value imputation", "Compression & denoising", "Feature generation"],
}

RECSYS = dict(
    id="recsys", name="Recommendation Systems", par="Recommender Systems",
    desc="Predict what a user will like: from item content, from similar users, or from learned latent factors.",
    groups=[
        ("Non-personalized", [("Popularity / Top-N", "Most bought / watched"),
                              ("Association Rules", "'Bought A → also B' (Apriori)")]),
        ("Content-based", [("Bag-of-Words / TF-IDF Item Profiles", "Describe items by their words"),
                           ("Cosine Similarity Ranking", "Items similar to what the user liked"),
                           ("Embedding Similarity", "Dense vectors (Word2Vec, BERT)")]),
        ("Collaborative Filtering", [("User-based Collaborative Filtering", "Users like you also liked…"),
                                     ("Item-based Collaborative Filtering", "Items rated alike by users"),
                                     ("Matrix Factorization (SVD / ALS)", "Latent user & item factors")]),
        ("Hybrid & Deep", [("Hybrid (weighted / switching)", "Mix content + collaborative"),
                           ("Neural CF / Two-Tower", "Learned user & item embeddings"),
                           ("Sequence-aware (RNN / Transformer)", "Next item in a session")]),
        ("Evaluation & Problems", [("Precision@K / Recall@K / NDCG", "Ranking quality"),
                                   ("Cold Start", "New user / new item, no history"),
                                   ("Sparsity & Popularity Bias", "Mostly empty matrix, rich get richer")]),
    ],
    pick="User-based Collaborative Filtering")

PROBLEMS = dict(
    id="problems", name="Common Data Problems", par="ML Workflow",
    desc="Problems that break models before the algorithm matters: dimensionality, imbalance, leakage, drift and more.",
    groups=[
        ("Shape of the Data", [("Curse of Dimensionality", "Space grows faster than data"),
                               ("Small Data (n small)", "High variance estimates"),
                               ("Multicollinearity", "Redundant, correlated features")]),
        ("Target Problems", [("Class Imbalance", "Rare positives (fraud, crashes)"),
                             ("Label Noise", "Wrong or inconsistent labels"),
                             ("Leaky Target", "Future / target info in features")]),
        ("Value Problems", [("Missing Data", "MCAR / MAR / MNAR"),
                            ("Outliers", "Extreme values distort fits"),
                            ("Skewed Features", "Long tails, heteroscedasticity")]),
        ("Time & Distribution", [("Non-stationarity", "Statistics change over time"),
                                 ("Data / Concept Drift", "Train ≠ production distribution"),
                                 ("Look-ahead / Survivorship Bias", "Backtest illusions")]),
        ("Imbalance Remedies", [("Resampling (Under / Over / SMOTE)", "Balance the training set"),
                                ("Class Weights / Cost-sensitive", "Penalise errors unequally"),
                                ("Threshold Moving & PR-AUC", "Decide and evaluate correctly")]),
    ],
    pick="Curse of Dimensionality")

BUILT_ON.update({
    "recsys": ["User–Item Matrix", "Similarity (cosine / Pearson)", "Neighbourhood (kNN)", "Weighted Average Prediction"],
    "problems": ["Data Size vs Features", "Class Distribution", "Train vs Production Data", "Validation Design"],
})
ASSUMPTIONS.update({
    "recsys": (["Users who agreed in the past agree in the future", "Enough overlapping ratings between users",
                "Preferences are stable over time", "Ratings are missing at random (often violated)"],
               "Matrix sparsity, overlap counts, offline Precision@K / NDCG",
               "Item-based CF, matrix factorization, content-based / hybrid for cold start"),
    "problems": (["Training data represents future data", "Features available at prediction time",
                  "Enough samples per feature and per class", "Labels are reliable"],
                 "Data profiling, class counts, n / p ratio, drift monitoring",
                 "Collect data, reduce features, resample / reweight, retrain on recent data"),
})
HIDDEN_USES.update({
    "recsys": ["Similar-asset discovery for portfolios", "News / research recommendation for traders",
               "Feature: 'users like this' scores in churn models", "Matrix factorization for missing-value imputation"],
    "problems": ["Data-quality checklists before modelling", "Monitoring dashboards in production",
                 "Explaining surprising backtest results", "Choosing models that tolerate the problem"],
})

PARADIGMS.insert(PARADIGMS.index("Foundations"), "Recommender Systems")
PARADIGMS.append("ML Workflow")
CATS = list(_RAW) + [DISTRIBUTIONS, STATISTICS, DATANATURE, HYPOTESTS, TSTESTS, LABELLING,
                     EVALUATION, FEATUREENG, EXPLAIN, FITTING, VERSATILE, RECSYS, PROBLEMS]
BY_ID = {c["id"]: c for c in CATS}

BODY_X = 400     # body (drawings) starts here; family column is x = 0 .. 340
BODY_Y = 240     # first panel y


def link(pid):
    return f"data:page/id,{pid}"


def header(pg: Page, cat, width=2400):
    """Title, breadcrumb, back link, family column and 'Built on' chain. Returns nothing."""
    pg.title(cat["pick"] if cat["id"] != "distributions" else cat["name"], 0, 0, 1400, 32)
    pg.text(f"{cat['par']}  ›  {cat['name']}  ›  ★ Base algorithm" if cat["id"] != "distributions"
            else f"{cat['par']}  ›  {cat['name']}", 0, 52, 1400, 20, 13, color="#555555")
    pg.text(cat["desc"], 0, 74, 1400, 20, 12, italic=True, color="#555555")
    pg.text("◀ Back to Overview", width - 200, 8, 200, 24, 13, bold=True, align="right", color=BLACK,
            link=link("overview"))
    if cat["id"] != "distributions":
        pg.text("Distributions & Statistics ▶", width - 260, 34, 260, 22, 11, align="right", color="#3333FF",
                link=link("distributions"))

    # family column
    pg.text("Algorithm Family", 0, 120, 340, 24, 15, bold=True)
    pg.line([(0, 146), (330, 146)], width=1)
    y = 156
    for gname, algs in cat["groups"]:
        pg.text(gname, 0, y, 340, 20, 12, bold=True)
        y += 22
        for name, desc in algs:
            base = name == cat["pick"]
            lab = (f'<span style="background:{HL}"><b>★ {E(name)}</b></span>' if base else E(name)) + \
                  f'<br><font color="#777777" style="font-size:9px">{E(desc)}</font>'
            pg.html(lab, 12, y, 320, 30, 11, valign="middle")
            pg.line([(4, y + 9), (9, y + 9)], width=0.75)
            y += 32
        y += 8

    # assumptions of the base algorithm
    if cat["id"] in ASSUMPTIONS:
        ass, check, fix = ASSUMPTIONS[cat["id"]]
        y += 10
        top = y
        pg.text(f"⚠ Assumptions of {cat['pick']}", 10, y + 6, 320, 34, 13, bold=True, valign="top")
        y += 42
        h = sum(15 * (1 + len(a) // 46) for a in ass) + 8
        pg.html('<ul style="margin:0;padding-left:16px">' + "".join(f"<li>{E(a)}</li>" for a in ass) + "</ul>",
                10, y, 320, h, 11)
        y += h + 4
        hc = 15 * (1 + len(check) // 44) + 4
        pg.html(f"<b>Check:</b> {E(check)}", 10, y, 320, hc, 11)
        y += hc + 6
        hf = 15 * (1 + len(fix) // 44) + 4
        pg.html(f'<b><font color="#FF0000">If violated →</font></b> {E(fix)}', 10, y, 320, hf, 11)
        y += hf + 10
        pg.rect(0, top, 340, y - top, stroke="#FF8000", dashed=True, sw=1)

    # versatile / hidden uses
    uses = globals().get("HIDDEN_USES", {}).get(cat["id"])
    if uses:
        y += 16
        top = y
        pg.text("💡 Also used for (hidden tricks)", 10, y + 6, 320, 22, 13, bold=True, color="#009900")
        y += 34
        h = sum(15 * (1 + len(u) // 46) for u in uses) + 8
        pg.html('<ul style="margin:0;padding-left:16px">' + "".join(f"<li>{E(u)}</li>" for u in uses) + "</ul>",
                10, y, 320, h, 11)
        y += h + 4
        pg.text("More: Versatile Uses page ▶", 10, y, 320, 18, 10, color="#3333FF", link=link("versatile"))
        y += 26
        pg.rect(0, top, 340, y - top, stroke="#009900", dashed=True, sw=1)

    # built-on chain
    items = BUILT_ON[cat["id"]]
    pg.text("Built on (background algorithms):", BODY_X, 112, 400, 20, 12, bold=True)
    x, ids = BODY_X, []
    for it in items:
        w = max(120, 7.2 * len(it) + 24)
        ids.append(pg.rect(x, 140, w, 40, it, size=12))
        x += w + 40
    base = pg.rect(x + 20, 134, max(200, 9 * len(cat["pick"]) + 40), 52, "★ " + cat["pick"], fill=HL, sw=2,
                   size=14, bold=True)
    for a, b in zip(ids, ids[1:] + [base]):
        pg.connect(a, b, orth=False)
    pg.line([(BODY_X, 206), (width, 206)], color="#BBBBBB", width=0.5)
