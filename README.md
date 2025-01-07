# Facial Recognition System for Access Control

## Project Overview

This project implements an intelligent facial recognition system for automated access control, combining both traditional machine learning (ML) and deep learning (DL) approaches to enhance security. The system identifies and verifies individuals in real-time, allowing authorized personnel to gain access while preventing unauthorized individuals from entering. It also provides clear feedback to security personnel, ensuring seamless access management.

### Business Problem

Organizations face the dual challenge of ensuring security and managing access effectively. Traditional methods such as key cards and manual ID checks are vulnerable to theft or forgery, and require human resources. The key challenges that this system addresses include:

- Accurately identifying authorized personnel in real-time.
- Preventing unauthorized access from known unwelcome individuals.
- Reducing manual effort for identity verification.
- Maintaining high throughput at access points without compromising security.
- Creating an audit trail of access attempts.

### Objective

1. Develop a reliable facial recognition system for automated access control.
2. Distinguish between welcome and unwelcome individuals in real-time.
3. Provide security personnel with clear, actionable feedback.

## Data Understanding

The dataset used in this project is **Labeled Faces in the Wild (LFW)**, which provides:

- 13,233 images of 5,749 different individuals.
- Real-world images captured with variations in lighting, pose, expression, background, and image quality.

**Data Quality Considerations**:

- The dataset reflects real-world conditions, making it suitable for practical applications.
- It includes demographic diversity and natural pose variations that pose challenges for the system.

## Methodology

The project is implemented in two approaches: traditional machine learning (ML) and deep learning (DL).

### 1. Data Preparation

- **Dataset**: The LFW dataset was fetched using `sklearn.datasets.fetch_lfw_people`.
- The dataset was cleaned and organized for training and testing.

### 2. Traditional ML Pipeline (PCA + SVM)

#### Dimensionality Reduction with PCA

- Principal Component Analysis (PCA) was applied for feature extraction, reducing the number of features to 150 eigenfaces.
- Images were projected onto these eigenfaces to extract features for classification.

#### SVM Classifier

- A Support Vector Machine (SVM) classifier with radial basis function (RBF) kernel was used to classify faces.
- Grid search was applied to find the optimal hyperparameters (`C` and `gamma`).

#### Evaluation

- The model was evaluated using the **confusion matrix** and **classification report** to assess the precision, recall, and accuracy.
- Key findings included high precision across all classes, although there were challenges with recall for certain classes like Gerhard Schroeder.

#### Results

- **Confusion Matrix**: The confusion matrix revealed minimal cross-class confusion with strong performance on certain classes (e.g., Colin Powell, George W. Bush).
- **Accuracy**: The model performed well in terms of precision but could improve in recall for specific individuals.

### 3. Deep Learning Approach (FaceNet)

#### FaceNet Model Integration

- The project leverages the **FaceNet** model to extract facial embeddings (encodings) for identification and verification.
- Pre-trained FaceNet model was used to convert faces into numerical representations (embeddings), enabling the comparison of individuals.

#### Face Verification and Recognition

- **Face Verification**: The system verifies if a given image matches an identity stored in the database by comparing embeddings using cosine similarity.
- **Face Recognition**: The system identifies individuals by comparing their embeddings with those in the database.

#### Database Creation

- A database of encodings was created for each person in the dataset by averaging their facial encodings.
- The database allows for quick retrieval and comparison when verifying or recognizing faces.

### 4. System Integration

- The face recognition system was integrated into an access control system that can classify known individuals and provide access while rejecting unknown individuals.
- For each incoming face, the system checks the database for a match and either grants or denies access based on the comparison result.

### 5. Evaluation

The system was tested using various test images, and performance was evaluated in terms of:

- **Accuracy**: How well the system identifies faces.
- **False Positives/Negatives**: How often the system misidentifies or rejects individuals.
- **Recall and Precision**: The ability to detect all instances of a class and avoid false positives.

## File Structure

```
/FacialRecognition
├── /data
│   └── lfw_dataset/        # Labeled Faces in the Wild dataset
├── /saved_model/           # Pre-trained FaceNet model
├── facial_recognition.py  # Main code for facial recognition system
└── README.md               # This README file
```

## Installation

To set up and run the facial recognition system, follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/facial-recognition-access-control.git
   cd facial-recognition-access-control
   ```

2. **Install Dependencies**:
   The project requires the following Python libraries:
   - `tensorflow`
   - `scikit-learn`
   - `matplotlib`
   - `numpy`
   - `seaborn`

   You can install the required packages using `pip`:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy seaborn
   ```

3. **Download the LFW Dataset**:
   You can fetch the LFW dataset by running the following Python script:
   ```python
   from sklearn.datasets import fetch_lfw_people
   lfw_dataset = fetch_lfw_people(min_faces_per_person=100)
   ```

4. **Run the System**:
   - You can execute the main script to test the facial recognition system:
     ```bash
     python facial_recognition.py
     ```

## Usage

Once the system is set up, the recognition and verification functionality can be tested using images of known and unknown individuals. The system will compare the facial encoding of the input image with the database and output whether access is granted or denied.

### Example:
```python
test_image = X[15]  # Select an image from the LFW dataset
verify(test_image, "Kofi Annan", database, FRmodel)  # Verify a known identity
```

- If the identity is recognized, the system outputs a welcome message.
- If the identity is not in the database, the system denies access.

## Results and Performance

- **Model Accuracy**: The system achieved strong classification performance with high precision across the classes.
- **FaceNet Approach**: Using the FaceNet model significantly improved the verification and recognition accuracy by leveraging facial embeddings.

## Future Improvements

- **Real-time Performance**: Optimize for real-time performance with faster face detection and recognition.
- **Expanded Database**: Include more individuals in the database for improved generalization.
- **Handling Variability**: Enhance the system's ability to handle challenging conditions such as low lighting and varying poses.

## References

- **Labeled Faces in the Wild**: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
- **FaceNet**: https://arxiv.org/abs/1503.03832

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
