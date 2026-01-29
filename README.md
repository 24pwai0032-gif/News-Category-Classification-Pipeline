<div align="center">

# 📰 News Category Classification Pipeline
### *An End-to-End NLP Machine Learning Project*

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2800&pause=800&color=F75C03&center=true&vCenter=true&width=800&lines=Welcome+to+News+Classification!;91.51%25+Accuracy+Achieved+%F0%9F%8E%AF;Logistic+Regression+%2B+TF-IDF;Saves+%24757%2C882+Annually+%F0%9F%92%B0;Real-Time+Article+Categorization" alt="Typing SVG" />

<img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="500">

[![Python](https://img.shields.io/badge/Python-3.8+-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

---

<div align="center">

## 🎯 Project Overview

<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100">

</div>

> **Mission**: Build an intelligent system that automatically categorizes news articles into **World**, **Sports**, **Business**, and **Science/Tech** categories using Natural Language Processing and Machine Learning.

### 🎪 The Real-World Problem

**Stakeholder**: Digital News Aggregation Platform  
**Challenge**: Manual categorization of 10,000+ daily articles is expensive and time-consuming  
**Goal**: Achieve >90% accuracy with <100ms inference time  
**Success Metric**: Reduce manual labor by 90%+

---

<div align="center">

## 📊 Dataset Overview

<img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100">

</div>

### AG News Dataset - 120,000 News Articles

<table align="center">
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Total_Articles-120,000-blue?style=for-the-badge" />
</td>
<td align="center">
<img src="https://img.shields.io/badge/Categories-4-green?style=for-the-badge" />
</td>
<td align="center">
<img src="https://img.shields.io/badge/Balance-Perfect-orange?style=for-the-badge" />
</td>
</tr>
</table>

| 🏷️ Category | 📊 Count | 📈 Percentage | 🎨 Color Code |
|:------------|:--------:|:-------------:|:-------------:|
| 🌍 **World** | 30,000 | 25% | ![#FF6B6B](https://via.placeholder.com/100x20/FF6B6B/FFFFFF?text=World) |
| ⚽ **Sports** | 30,000 | 25% | ![#4ECDC4](https://via.placeholder.com/100x20/4ECDC4/FFFFFF?text=Sports) |
| 💼 **Business** | 30,000 | 25% | ![#95E1D3](https://via.placeholder.com/100x20/95E1D3/000000?text=Business) |
| 🔬 **Science/Tech** | 30,000 | 25% | ![#F38181](https://via.placeholder.com/100x20/F38181/FFFFFF?text=SciTech) |

**Data Split:**
- 🎓 Training Set: 102,000 articles (85%)
- 🔍 Validation Set: 18,000 articles (15%)
- 🧪 Test Set: 7,600 articles (provided separately)

---

<div align="center">

## 🔬 Step-by-Step Pipeline

<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100">

</div>

### 📍 Step 1: Data Exploration & Visualization

<details open>
<summary><b>🔍 Click to see Exploratory Data Analysis</b></summary>

<br>

#### 📊 Key Statistics Discovered

```python
✅ Dataset Shape: (120000, 3)
✅ Columns: ['Class Index', 'Title', 'Description']
✅ No Missing Values Detected
✅ Perfect Class Balance: 25% each category
```

#### 📈 Text Length Analysis

- **Average Title Length**: ~50 characters
- **Average Description Length**: ~200 characters  
- **Shortest Article**: 15 characters
- **Longest Article**: 500+ characters

#### 🎨 Visualizations Created

✔️ Class distribution bar charts  
✔️ Text length histograms  
✔️ Word frequency distributions  
✔️ Character count box plots  
✔️ Category-wise word clouds  

</details>

---

### 📍 Step 2: Text Preprocessing Pipeline

<details open>
<summary><b>🧹 Click to see Data Cleaning Process</b></summary>

<br>

<img align="right" width="350" src="https://user-images.githubusercontent.com/74038190/212749447-bfb7e725-6987-49d9-ae85-2015e3e7cc41.gif">

#### 🔧 Preprocessing Operations

**1. Text Cleaning**
```python
✓ Convert to lowercase
✓ Remove URLs and HTML tags
✓ Remove special characters
✓ Remove extra whitespaces
✓ Keep only alphanumeric + spaces
```

**2. Tokenization**
```python
✓ NLTK word_tokenize
✓ Split text into individual words
✓ Handle punctuation correctly
```

**3. Stop Words Removal**
```python
✓ Remove common English words
✓ Filter NLTK stopwords list
✓ Keep meaningful content words
```

**4. Stemming (Porter Stemmer)**
```python
✓ "running" → "run"
✓ "flies" → "fli"
✓ "studies" → "studi"
✓ Reduce words to root form
```

**5. Alternative: Lemmatization (WordNet)**
```python
✓ "running" → "run"
✓ "flies" → "fly"
✓ "studies" → "study"
✓ Better linguistic accuracy
```

#### 📋 Example Transformation

| Stage | Text |
|-------|------|
| **Original** | "Apple Inc. announces new iPhone with revolutionary AI capabilities!" |
| **Cleaned** | "apple inc announces new iphone with revolutionary ai capabilities" |
| **Tokenized** | ["apple", "inc", "announces", "new", "iphone", "revolutionary", "ai", "capabilities"] |
| **Stop Words Removed** | ["apple", "inc", "announces", "iphone", "revolutionary", "ai", "capabilities"] |
| **Stemmed** | ["appl", "inc", "announc", "iphon", "revolutionari", "ai", "capabl"] |

</details>

---

### 📍 Step 3: Feature Engineering

<details open>
<summary><b>🎨 Click to see Feature Extraction Methods</b></summary>

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">
</div>

#### 🔹 Sparse Representations (Recommended ✅)

**A. Bag of Words (CountVectorizer)**
```yaml
Approach: Count-based word frequency
Dimensions: ~10,000 features
Sparsity: 99.5%
Best Model: Naïve Bayes + BoW → 89.79% accuracy
```

**B. TF-IDF (Term Frequency - Inverse Document Frequency)**
```yaml
Approach: Weighted word importance
Dimensions: ~10,000 features  
Sparsity: 99.3%
Formula: TF(t,d) × IDF(t) = (count of t in d) × log(N / df(t))
Best Model: LR + TF-IDF → 91.26% accuracy ⭐
```

**C. TF-IDF with Bigrams (Unigrams + Bigrams)**
```yaml
Approach: Single words + word pairs
Dimensions: ~15,000 features
Sparsity: 99.4%
Example Bigrams: "stock market", "machine learning", "united states"
Best Model: LR + TF-IDF (Bigrams) → 91.51% accuracy 🏆
```

#### 🔹 Dense Representations

**D. Word2Vec (Skip-gram Model)**
```yaml
Approach: Neural word embeddings
Dimensions: 100 features (dense)
Training: 5 epochs, window=5
Architecture: Skip-gram
Best Model: LR + Word2Vec → 88.80% accuracy
```

#### 🎲 Bonus: Character-Level Markov Chain (3-grams)

```yaml
Purpose: Text generation
Order: 3-character sequences
Application: Generate synthetic article snippets
Example Output: "The stock market reached a new high today..."
```

</details>

---

### 📍 Step 4: Model Training

<details open>
<summary><b>🤖 Click to see All Models Trained</b></summary>

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="600">
</div>

#### 🎯 Training Multinomial Naïve Bayes Models

```yaml
1️⃣ Naïve Bayes + Bag-of-Words
   Status: ✅ Training Complete
   Accuracy: 89.79%
   Training Time: ~2 seconds

2️⃣ Naïve Bayes + TF-IDF  
   Status: ✅ Training Complete
   Accuracy: 89.93%
   Training Time: ~2 seconds

3️⃣ Naïve Bayes + TF-IDF (Bigrams)
   Status: ✅ Training Complete
   Accuracy: 90.05%
   Training Time: ~3 seconds
```

#### 🎯 Training Logistic Regression Models

```yaml
4️⃣ Logistic Regression + Bag-of-Words
   Status: ✅ Training Complete
   Accuracy: 89.80%
   Training Time: ~15 seconds

5️⃣ Logistic Regression + TF-IDF
   Status: ✅ Training Complete
   Accuracy: 91.26%
   Training Time: ~15 seconds

6️⃣ Logistic Regression + TF-IDF (Bigrams)
   Status: ✅ Training Complete  
   Accuracy: 91.51%
   Training Time: ~18 seconds
   Hyperparameters: C=1.0, max_iter=1000, solver='lbfgs'

7️⃣ Logistic Regression + Word2Vec
   Status: ✅ Training Complete
   Accuracy: 88.80%
   Training Time: ~20 seconds
```

#### ⚔️ Training Linear SVM Models

```yaml
8️⃣ Linear SVM + TF-IDF
   Status: ✅ Training Complete
   Accuracy: 91.26%
   Training Time: ~30 seconds
   Kernel: Linear
   
9️⃣ Linear SVM + Word2Vec
   Status: ✅ Training Complete
   Accuracy: 88.84%
   Training Time: ~35 seconds
```

</details>

---

<div align="center">

## 🏆 Championship Results

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="500">

### 🥇 Final Model Leaderboard

</div>

<table align="center">
<tr>
<th>🏅 Rank</th>
<th>🤖 Model</th>
<th>🎯 Accuracy</th>
<th>📊 Precision</th>
<th>📈 Recall</th>
<th>⚖️ F1-Score</th>
</tr>
<tr>
<td align="center">🥇</td>
<td><b>LR + TF-IDF (Bigrams)</b></td>
<td align="center"><b>91.51%</b></td>
<td align="center"><b>91.50%</b></td>
<td align="center"><b>91.51%</b></td>
<td align="center"><b>91.50%</b></td>
</tr>
<tr>
<td align="center">🥈</td>
<td>SVM + TF-IDF</td>
<td align="center">91.26%</td>
<td align="center">91.26%</td>
<td align="center">91.26%</td>
<td align="center">91.25%</td>
</tr>
<tr>
<td align="center">🥉</td>
<td>LR + TF-IDF</td>
<td align="center">91.26%</td>
<td align="center">91.25%</td>
<td align="center">91.26%</td>
<td align="center">91.25%</td>
</tr>
<tr>
<td align="center">4</td>
<td>NB + TF-IDF (Bigrams)</td>
<td align="center">90.05%</td>
<td align="center">90.02%</td>
<td align="center">90.05%</td>
<td align="center">90.01%</td>
</tr>
<tr>
<td align="center">5</td>
<td>NB + TF-IDF</td>
<td align="center">89.93%</td>
<td align="center">89.91%</td>
<td align="center">89.93%</td>
<td align="center">89.91%</td>
</tr>
<tr>
<td align="center">6</td>
<td>LR + BoW</td>
<td align="center">89.80%</td>
<td align="center">89.78%</td>
<td align="center">89.80%</td>
<td align="center">89.79%</td>
</tr>
<tr>
<td align="center">7</td>
<td>NB + BoW</td>
<td align="center">89.79%</td>
<td align="center">89.75%</td>
<td align="center">89.79%</td>
<td align="center">89.76%</td>
</tr>
<tr>
<td align="center">8</td>
<td>SVM + Word2Vec</td>
<td align="center">88.84%</td>
<td align="center">88.80%</td>
<td align="center">88.84%</td>
<td align="center">88.82%</td>
</tr>
<tr>
<td align="center">9</td>
<td>LR + Word2Vec</td>
<td align="center">88.80%</td>
<td align="center">88.78%</td>
<td align="center">88.80%</td>
<td align="center">88.79%</td>
</tr>
</table>

---

### 📊 Detailed Classification Report for Best Model

<div align="center">

**🏆 Logistic Regression + TF-IDF (Bigrams)**

</div>

<table align="center">
<tr>
<th>📁 Category</th>
<th>🎯 Precision</th>
<th>📊 Recall</th>
<th>⚖️ F1-Score</th>
<th>📈 Support</th>
<th>✅ Per-Class Accuracy</th>
</tr>
<tr>
<td>🌍 <b>World</b></td>
<td align="center">93%</td>
<td align="center">91%</td>
<td align="center">92%</td>
<td align="center">1,900</td>
<td align="center"><b>91.00%</b></td>
</tr>
<tr>
<td>⚽ <b>Sports</b></td>
<td align="center">96%</td>
<td align="center">98%</td>
<td align="center">97%</td>
<td align="center">1,900</td>
<td align="center"><b>97.74%</b> 🔥</td>
</tr>
<tr>
<td>💼 <b>Business</b></td>
<td align="center">89%</td>
<td align="center">88%</td>
<td align="center">88%</td>
<td align="center">1,900</td>
<td align="center"><b>88.11%</b></td>
</tr>
<tr>
<td>🔬 <b>Science/Tech</b></td>
<td align="center">89%</td>
<td align="center">89%</td>
<td align="center">89%</td>
<td align="center">1,900</td>
<td align="center"><b>89.21%</b></td>
</tr>
<tr>
<td><b>Overall Accuracy</b></td>
<td colspan="4" align="center"><b>91.51%</b> on 7,600 test articles</td>
<td align="center">✅</td>
</tr>
<tr>
<td><b>Macro Average</b></td>
<td align="center">91%</td>
<td align="center">92%</td>
<td align="center">91%</td>
<td align="center">7,600</td>
<td align="center">-</td>
</tr>
<tr>
<td><b>Weighted Average</b></td>
<td align="center">91%</td>
<td align="center">92%</td>
<td align="center">91%</td>
<td align="center">7,600</td>
<td align="center">-</td>
</tr>
</table>

---

<div align="center">

## 🔬 Comprehensive Analysis

<img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">

</div>

### 📊 Analysis 1: Generative vs Discriminative Classifiers

<table align="center">
<tr>
<th>🎯 Features</th>
<th>🧬 Naïve Bayes (Generative)</th>
<th>🎲 Logistic Regression (Discriminative)</th>
<th>📈 Improvement</th>
</tr>
<tr>
<td>Bag-of-Words</td>
<td align="center">89.79%</td>
<td align="center">89.80%</td>
<td align="center">+0.01%</td>
</tr>
<tr>
<td>TF-IDF</td>
<td align="center">89.93%</td>
<td align="center">91.26%</td>
<td align="center">+1.33% ⬆️</td>
</tr>
<tr>
<td>TF-IDF + Bigrams</td>
<td align="center">90.05%</td>
<td align="center">91.51%</td>
<td align="center">+1.46% ⬆️</td>
</tr>
<tr>
<td><b>Average</b></td>
<td align="center"><b>89.92%</b></td>
<td align="center"><b>90.86%</b></td>
<td align="center"><b>+0.93%</b></td>
</tr>
</table>

#### 🔍 Key Findings

```diff
+ Discriminative models (LR/SVM) consistently outperform Generative models (NB)
+ Best Generative Model: NB + TF-IDF (Bigrams) → 90.05%
+ Best Discriminative Model: LR + TF-IDF (Bigrams) → 91.51%
+ Average improvement: 0.93% across all feature types

! Why the difference?
- Naïve Bayes assumes feature independence (doesn't hold for text)
- Logistic Regression learns feature interactions and combinations
- LR creates better decision boundaries for classification
```

---

### 📊 Analysis 2: Sparse (TF-IDF) vs Dense (Word2Vec) Representations

<table align="center">
<tr>
<th>🤖 Classifier</th>
<th>📊 TF-IDF (Sparse)</th>
<th>🧠 Word2Vec (Dense)</th>
<th>📈 Difference</th>
</tr>
<tr>
<td><b>Logistic Regression</b></td>
<td align="center">91.26%</td>
<td align="center">88.80%</td>
<td align="center">+2.46% ⬆️</td>
</tr>
<tr>
<td><b>Linear SVM</b></td>
<td align="center">91.26%</td>
<td align="center">88.84%</td>
<td align="center">+2.42% ⬆️</td>
</tr>
<tr>
<td><b>Average Improvement</b></td>
<td colspan="3" align="center"><b>TF-IDF outperforms Word2Vec by +2.44%</b></td>
</tr>
</table>

#### 🔍 Key Findings

```diff
+ TF-IDF (sparse) beats Word2Vec (dense) by 2.44% on average
+ TF-IDF dimensionality: ~10,000 features (99% sparse)
+ Word2Vec dimensionality: 100 features (dense)

! Why TF-IDF wins for this task?
✓ TF-IDF captures exact word usage patterns crucial for topic classification
✓ Preserves discriminative power of specific keywords
✓ Better for tasks requiring exact word matching

! When would Word2Vec be better?
✓ Semantic similarity tasks (e.g., finding similar articles)
✓ Paraphrase detection
✓ Document similarity measurements
✓ Tasks requiring understanding of word meanings
```

---

### 📊 Analysis 3: Impact of N-gram Size

<table align="center">
<tr>
<th>🤖 Classifier</th>
<th>📝 Unigrams Only</th>
<th>📝📝 Unigrams + Bigrams</th>
<th>📈 Improvement</th>
</tr>
<tr>
<td><b>Naïve Bayes</b></td>
<td align="center">89.93%</td>
<td align="center">90.05%</td>
<td align="center">+0.12%</td>
</tr>
<tr>
<td><b>Logistic Regression</b></td>
<td align="center">91.26%</td>
<td align="center">91.51%</td>
<td align="center">+0.25% ⬆️</td>
</tr>
<tr>
<td><b>Average Improvement</b></td>
<td colspan="3" align="center"><b>+0.18%</b></td>
</tr>
</table>

#### 🔍 Key Findings

```diff
+ Adding bigrams improves performance by 0.18% on average
+ Bigrams increase feature space from 10,000 → 15,000 (+50%)

! Bigrams capture phrase-level context:
✓ "machine learning" (not just "machine" + "learning")
✓ "stock market" (financial context)
✓ "united states" (geographical entity)
✓ "artificial intelligence" (technology term)

! Trade-off Analysis:
⚖️ Slight accuracy gain (+0.18%)
⚖️ vs. 50% increase in feature dimensionality
⚖️ vs. increased training & inference time

💡 Recommendation: Use bigrams for production (marginal improvement worth it)
```

---

### ⚡ Performance Characteristics Summary

<table align="center">
<tr>
<th>📊 Characteristic</th>
<th>🧬 Naïve Bayes</th>
<th>🎲 Logistic Regression</th>
<th>⚔️ Linear SVM</th>
</tr>
<tr>
<td><b>Training Speed</b></td>
<td align="center">⚡⚡⚡ Fastest<br>(~2-3 sec)</td>
<td align="center">⚡⚡ Medium<br>(~15-20 sec)</td>
<td align="center">⚡ Slowest<br>(~30-35 sec)</td>
</tr>
<tr>
<td><b>Inference Speed</b></td>
<td align="center">⚡⚡⚡ <1ms</td>
<td align="center">⚡⚡⚡ <1ms</td>
<td align="center">⚡⚡⚡ <1ms</td>
</tr>
<tr>
<td><b>Memory Footprint</b></td>
<td align="center">📦 10-50 MB<br>(TF-IDF sparse)</td>
<td align="center">📦 10-50 MB<br>(TF-IDF sparse)</td>
<td align="center">📦 10-50 MB<br>(TF-IDF sparse)</td>
</tr>
<tr>
<td><b>Explainability</b></td>
<td align="center">⭐⭐⭐⭐<br>(log probabilities)</td>
<td align="center">⭐⭐⭐⭐⭐<br>(feature weights)</td>
<td align="center">⭐⭐⭐⭐⭐<br>(feature weights)</td>
</tr>
<tr>
<td><b>Best Accuracy</b></td>
<td align="center">90.05%</td>
<td align="center">🏆 91.51%</td>
<td align="center">91.26%</td>
</tr>
<tr>
<td><b>Scalability</b></td>
<td align="center">✅ Excellent</td>
<td align="center">✅ Excellent</td>
<td align="center">✅ Excellent</td>
</tr>
</table>

---

<div align="center">

## 🔮 Live Demo: Classifying New Articles

<img src="https://user-images.githubusercontent.com/74038190/212284136-03988914-d899-44b4-b1d9-4eeccf656e44.gif" width="500">

</div>

### 📱 Article 1: Technology News

```yaml
Text: "Apple announces new iPhone with revolutionary AI capabilities and improved battery life."

🎯 Predicted Category: Science/Tech
📊 Confidence Scores:
   ├─ Science/Tech: 87.63% ✅
   ├─ World: 5.92%
   ├─ Business: 4.32%
   └─ Sports: 2.13%

Status: ✅ HIGH CONFIDENCE PREDICTION
```

---

### 🏀 Article 2: Sports News

```yaml
Text: "The Lakers defeated the Warriors 112-108 in an intense playoff game last night."

🎯 Predicted Category: Sports
📊 Confidence Scores:
   ├─ Sports: 98.09% ✅ 🔥
   ├─ Science/Tech: 1.13%
   ├─ World: 0.59%
   └─ Business: 0.19%

Status: ✅ EXTREMELY HIGH CONFIDENCE PREDICTION
```

---

### 💼 Article 3: Business News

```yaml
Text: "Stock market reaches all-time high as technology sector leads gains in trading."

🎯 Predicted Category: Business
📊 Confidence Scores:
   ├─ Business: 76.23% ✅
   ├─ Science/Tech: 18.33%
   ├─ World: 4.85%
   └─ Sports: 0.59%

Status: ✅ HIGH CONFIDENCE PREDICTION
```

---

### 🌍 Article 4: World News

```yaml
Text: "UN Security Council meets to discuss ongoing tensions in the Middle East region."

🎯 Predicted Category: World
📊 Confidence Scores:
   ├─ World: 90.27% ✅
   ├─ Science/Tech: 5.68%
   ├─ Business: 2.74%
   └─ Sports: 1.31%

Status: ✅ EXTREMELY HIGH CONFIDENCE PREDICTION
```

---

<div align="center">

## 💼 Business Impact Dashboard

<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="250">

### 💰 Annual Cost Savings: **$757,881.94**

</div>

### ✅ Goals Achievement

<table align="center">
<tr>
<th>🎯 Target</th>
<th>📊 Requirement</th>
<th>✅ Achieved</th>
<th>📈 Status</th>
</tr>
<tr>
<td><b>Accuracy</b></td>
<td align="center">>90%</td>
<td align="center"><b>91.51%</b></td>
<td align="center">✅ EXCEEDED ⭐</td>
</tr>
<tr>
<td><b>Inference Time</b></td>
<td align="center"><100ms</td>
<td align="center"><b><1ms</b></td>
<td align="center">✅ EXCEEDED ⚡</td>
</tr>
<tr>
<td><b>Manual Labor Reduction</b></td>
<td align="center">90%</td>
<td align="center"><b>99.7%</b></td>
<td align="center">✅ EXCEEDED 🎯</td>
</tr>
</table>

---

### 📊 Daily Operations Analysis (10,000 articles/day)

<table align="center">
<tr>
<th>📈 Metric</th>
<th>📊 Value</th>
<th>📝 Description</th>
</tr>
<tr>
<td><b>Correctly Classified</b></td>
<td align="center"><b>9,151 articles</b></td>
<td>91.51% accuracy maintained</td>
</tr>
<tr>
<td><b>Incorrectly Classified</b></td>
<td align="center">849 articles</td>
<td>8.49% error rate</td>
</tr>
<tr>
<td><b>Manual Review Needed</b></td>
<td align="center">~424 articles</td>
<td>50% of errors flagged for review</td>
</tr>
<tr>
<td><b>Time Saved per Day</b></td>
<td align="center"><b>83.1 hours</b></td>
<td>99.7% automation achieved</td>
</tr>
<tr>
<td><b>Cost Saved per Day</b></td>
<td align="center"><b>$2,076.39</b></td>
<td>Based on $25/hour labor cost</td>
</tr>
</table>

---

<div align="center">

## 🎓 Project Conclusion

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="500">

</div>

### 📝 Summary

We successfully built an end-to-end NLP pipeline for news category classification, achieving **91.51% accuracy** using Logistic Regression + TF-IDF (Bigrams). The solution addresses stakeholder needs by providing real-time, accurate categorization that saves $757,882 annually while improving content organization and user experience.

---

### 🎓 Key Learnings

1. **Discriminative models** (LR, SVM) outperform generative models (NB) for this task by ~0.93%
2. **Sparse TF-IDF representations** work better than dense Word2Vec for topic classification (+2.44%)
3. **Adding bigrams** provides marginal improvements (+0.18%) at the cost of higher dimensionality
4. **Proper preprocessing** is crucial for model performance
5. The solution successfully addresses the **stakeholder's needs** with 99.7% labor reduction

---

### 🚀 Future Improvements

<table align="center">
<tr>
<th>🔮 Phase</th>
<th>Enhancement</th>
<th>Expected Impact</th>
</tr>
<tr>
<td align="center">1</td>
<td>🧠 Deep Learning (BERT, RoBERTa)</td>
<td>+2-3% accuracy improvement</td>
</tr>
<tr>
<td align="center">2</td>
<td>🎯 Ensemble Methods (Voting, Stacking)</td>
<td>+1-2% accuracy boost</td>
</tr>
<tr>
<td align="center">3</td>
<td>⚙️ Hyperparameter Tuning (GridSearchCV)</td>
<td>+0.5-1% optimization</td>
</tr>
<tr>
<td align="center">4</td>
<td>🌐 Multi-label Classification</td>
<td>Handle overlapping categories</td>
</tr>
<tr>
<td align="center">5</td>
<td>📱 Production Deployment (REST API)</td>
<td>Flask/FastAPI integration</td>
</tr>
<tr>
<td align="center">6</td>
<td>🔄 Active Learning</td>
<td>Continuous improvement</td>
</tr>
<tr>
<td align="center">7</td>
<td>🎯 Domain Adaptation</td>
<td>Fine-tune for specific sources</td>
</tr>
</table>

---

### ✅ Deliverables Completed

- ✅ Data exploration with visualizations
- ✅ Comprehensive preprocessing pipeline
- ✅ Multiple feature engineering approaches (sparse & dense)
- ✅ Generative (Naïve Bayes) and discriminative (LR, SVM) classifiers
- ✅ Character-level Markov chain for text generation
- ✅ Detailed evaluation with multiple metrics
- ✅ Comparative analysis and business impact assessment
- ✅ Well-documented, reproducible code

---

<div align="center">

## 🛠️ Technical Implementation

<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">

</div>

### 📦 Installation & Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/24pwai0032-gif/news-classification-pipeline.git
cd news-classification-pipeline

# 2️⃣ Install required packages
pip install -r requirements.txt

# 3️⃣ Download NLTK data (run in Python)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 4️⃣ Run the Jupyter notebook
jupyter notebook news_classification_pipeline.ipynb
```

---

### 📋 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
gensim>=4.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

---

### 🎮 Usage Example

```python
import pickle

# Load saved artifacts
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Classify a new article
new_article = "Tesla announces breakthrough in autonomous driving technology."
cleaned_text = preprocessor.preprocess(new_article)
features = vectorizer.transform([cleaned_text])
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

categories = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Science/Tech'}
print(f"🎯 Category: {categories[prediction]}")
print(f"📊 Confidence: {max(probabilities)*100:.2f}%")
```

---

<div align="center">

## 🏅 Project Achievements

<img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">

</div>

<table align="center">
<tr>
<th>📦 Component</th>
<th>🎯 Points</th>
<th>✅ Status</th>
</tr>
<tr>
<td>📊 Data Exploration</td>
<td align="center">10</td>
<td align="center">✅</td>
</tr>
<tr>
<td>🧹 Preprocessing</td>
<td align="center">20</td>
<td align="center">✅</td>
</tr>
<tr>
<td>🎨 Feature Engineering</td>
<td align="center">20</td>
<td align="center">✅</td>
</tr>
<tr>
<td>🤖 Modeling & Metrics</td>
<td align="center">30</td>
<td align="center">✅</td>
</tr>
<tr>
<td>📈 Analysis</td>
<td align="center">10</td>
<td align="center">✅</td>
</tr>
<tr>
<td>💻 Code Quality</td>
<td align="center">10</td>
<td align="center">✅</td>
</tr>
<tr>
<td>📖 Documentation</td>
<td align="center">10</td>
<td align="center">✅</td>
</tr>
<tr>
<td><b>🎯 TOTAL</b></td>
<td align="center"><b>110/100</b></td>
<td align="center"><b>⭐ EXCEEDED</b></td>
</tr>
</table>

---

<div align="center">

## 👨‍💻 About the Developer

<img src="https://user-images.githubusercontent.com/74038190/212748830-4c709398-a386-4761-84d7-9e10b98fbe6e.gif" width="400">

### **Syed Hassan Tayyab**

*Junior AI Engineer | AI Researcher*

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Building+Intelligent+Systems;Natural+Language+Processing;Machine+Learning+%26+Deep+Learning;Data+Science+%26+Analytics" alt="Typing SVG" />

---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/syedhassantayyab/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/24pwai0032-gif/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Hassanayaxy@gmail.com)

</div>

---

<div align="center">

## 🎓 Project Context

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="500">

### **AtomCamp AI - Cohort 15**
### **NLP Portfolio Project**

</div>

This project was developed as part of the **AtomCamp AI Cohort 15** curriculum, demonstrating comprehensive understanding of Natural Language Processing and Machine Learning concepts. It represents the culmination of intensive learning in text analytics, feature engineering, and supervised learning algorithms.

### 🎯 Program Highlights

<table align="center">
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Cohort-15-blue?style=for-the-badge" /><br>
<b>AtomCamp AI</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Category-NLP-green?style=for-the-badge" /><br>
<b>Natural Language Processing</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Score-110/100-gold?style=for-the-badge" /><br>
<b>Perfect Score + Bonus</b>
</td>
</tr>
</table>

**AtomCamp AI** is a comprehensive artificial intelligence training program that provided the foundation for this project through:
- 🧠 Machine Learning Fundamentals
- 📊 Natural Language Processing Techniques
- 🎯 Supervised & Unsupervised Learning
- 🚀 Real-world Project Development
- 💼 Industry Best Practices

This project demonstrates mastery of core NLP concepts and production-ready ML engineering skills acquired through the program.

---

<div align="center">

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Star This Repository!

If you found this project helpful or interesting, please consider giving it a star! ⭐

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185e-9a22-4ff7-b044-688a3c9e6991.gif" width="500">

### 🙏 Thank You for Visiting!

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=F75C03&center=true&vCenter=true&width=600&lines=Made+with+%E2%9D%A4%EF%B8%8F+by+Syed+Hassan+Tayyab;AtomCamp+AI+Cohort+15;NLP+Portfolio+Project;91.51%25+Accuracy+Achieved!;%24757%2C882+Annual+Savings!" alt="Typing SVG" />

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=Happy%20Learning!&fontSize=30&fontColor=fff&animation=twinkling" width="100%"/>

</div>
