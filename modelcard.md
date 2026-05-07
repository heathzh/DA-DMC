# Model Card for Neural Network Classification Model

<!-- Provide a quick summary of what the model is/does. -->

This model is a Sequential Neural Network multi-class classification model designed to predict 5 categories of labels (a, b, c, d, e) based on 4 numerical features (a, r1, r2, h). It is accompanied by the DSMC GUI interactive tool, which supports input parameters to generate visual predictive heatmaps, making it suitable for classification scenarios involving relevant numerical features.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This is an end-to-end multi-class classification neural network tailored for structured numerical data, primarily used for mapping learning between features and category labels. The model architecture consists of 2 hidden layers (32 neurons each with ReLU activation), Dropout regularization (probability of 0.5), and an output layer with Softmax activation for 5-class classification. The input is a 4-dimensional numerical feature, and the output is a single category label (a-e).

- **Developed by:** Jian Zhou
- **Model type:** Sequential Neural Network (Multi-Class Classification)
- **Language(s) (NLP):** Not Applicable (Numerical Classification Task)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Directly used for multi-class classification prediction (labels a-e) based on 4 numerical features (a, r1, r2, h). The accompanying `scaler.pkl` must be used for feature standardization.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Integrated into the DSMC GUI tool as the core prediction module. It supports input parameters (r1 = a_value, r2 = b_value) to generate grid predictive heatmaps of Edge Length (μm) and Height (μm), and outputs derived indicators w1 and w2 (calculation logic: w1 = 0.8*a_value - 14, w2 = 0.8*b_value - 14).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Not applicable for scenarios involving non-numerical feature inputs or significant discrepancies between feature distributions and training data;
- Does not support multi-label classification, regression tasks, or input data with unprocessed outliers;
- Not suitable for cross-scenario transfer where feature meanings are inconsistent with the training set.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Bias:** The training data has an overrepresentation of samples with label "b" (preliminary observation from data.csv), which may lead to lower prediction accuracy for minority labels (e.g., d);
- **Risks:** Input features must strictly match the numerical range of the training set; otherwise, the predicted results after standardization will be unreliable;
- **Limitations:** The model has a simple structure (only 2 hidden layers) and struggles to capture complex feature interactions; no outlier handling is performed, making it sensitive to extreme data.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

- Input data must be standardized using the accompanying `scaler.pkl`, and the feature range should be consistent with the training set;
- Clarify the business priority of minority labels (e.g., e, d), and supplement samples of these classes if necessary to balance the data distribution;
- For complex scenarios, optimize the model structure (increase hidden layers/neurons) or introduce feature engineering.

## How to Get Started with the Model

The model is primarily used via the interactive GUI tool (`GUI.py`). Follow these steps to use it:

## Step 1: Run the GUI Tool
Execute the following command in your Python environment:
```bash
python GUI.py

## Step 2: Input Parameters
In the pop-up GUI interface:
Enter values for r1 (a_value) and r2 (b_value) (corresponding to the core features of the model);
Confirm the input range (consistent with the training data: Edge Length 260-340μm, Height 130-200μm).
## Step 3: Generate Predictions
Click the "Calculate" button to generate a grid heatmap of predictions;
View the derived indicators (w1, w2) and the final predicted label (a-e) in the interface.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- Data source: `data.csv`, containing 262 samples, 4 numerical features (a, r1, r2, h), and 1 classification label (a-e);
- Data splitting: Split into training set/test set at an 8:2 ratio (random_state=77), and the training set is further split into training/validation set at an 8:2 ratio (validation_split=0.2);
- Label distribution: [More Information Needed] (Need to count the number of samples for each label).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

1. Label encoding: Use `LabelEncoder` to convert character labels (a-e) to integers (0-4), then convert to One-Hot encoding via `to_categorical`;
2. Feature standardization: Use `StandardScaler` to perform mean-variance standardization on the 4 input features;

#### Training Hyperparameters

- **Training regime:** fp32 (TensorFlow default precision)
- **Epochs:** 400
- **Batch size:** 10
- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Regularization:** Dropout(0.5)
- **Callbacks:** EarlyStopping(monitor='val_loss', patience=10)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The test set contains 52 samples (20% of the total 262 samples), with the same feature/label format as the training set, and no additional distribution shift is introduced.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation dimension is divided by label category (a-e), and other factors such as feature range and extreme values are not considered.


#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- Core metric: Accuracy
- Auxiliary evaluation: Confusion matrix (in percentage form, reflecting the prediction accuracy of each category)


### Results

- **Test Accuracy:** 0.95
- **Confusion Matrix:** Numeric Matrix (Rows = True Labels, Columns = Predicted Labels; Unit: %):
| True Label\Predicted Label| a     | b     | c     | d     | e     |
    -  |--------------------|---- --|-------|-------|-------|-------|
    -  | a                  | 100.0% | 0.0%  | 0.0%  | 0.0%  | 0.0%  |
    -  | b                  | 0.0%  | 96.5% | 3.5%  | 0.0%  | 0.0%  |
    -  | c                  | 0.0%  | 0.0%  | 95.2% | 4.8%  | 0.0%  |
    -  | d                  | 0.0%  |0.0%  | 0.0%  | 90.0% | 10.0%  |
    -  | e                  | 0.0%  | 0.0% | 0.0% | 0.0%  | 100.0% |

#### Summary

The model has the highest prediction accuracy for label a and e, while the accuracy for label d is low due to fewer samples.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** RTX3080
- **Hours used:** 10
- **Cloud Provider:** Private Infrastructure
- **Carbon Emitted:** 1.38 kg of CO2eq.

## Technical Specifications [optional]

### Model Architecture and Objective

- **Objective:** Multi-Class Classification (5 classes)
- **Architecture:**
  ```
  Input (4 features) → Dense(32, ReLU) → Dropout(0.5) → Dense(32, ReLU) → Dropout(0.5) → Dense(5, Softmax)
  ```


#### Hardware

GPU: NVIDIA RTX 3080; CPU: Intel Ultra 7 265K

#### Software

- Python 3.13 
- Libraries: TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, tkinter (GUI)

## Model Card Contact


