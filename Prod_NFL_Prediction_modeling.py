{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
import matplotlib.pyplot as plt\
from scipy import stats\
import missingno as msno\
import seaborn as sns\
from sklearn.linear_model import LogisticRegression\
from sklearn.model_selection import train_test_split, RandomizedSearchCV\
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, confusion_matrix, classification_report\
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC\
from xgboost import XGBClassifier\
import xgboost as xgb\
import pickle\
import time\
import warnings\
import smtplib\
from email.mime.text import MIMEText\
\
warnings.filterwarnings('ignore')\
\
def send_email(subject, body, to_email):\
    from_email = "your_email@example.com"\
    password = "your_password"\
    msg = MIMEText(body)\
    msg['Subject'] = subject\
    msg['From'] = from_email\
    msg['To'] = to_email\
\
    try:\
        server = smtplib.SMTP_SSL('smtp.example.com', 465)\
        server.login(from_email, password)\
        server.sendmail(from_email, [to_email], msg.as_string())\
        server.quit()\
        print("Email sent successfully!")\
    except Exception as e:\
        print(f"Failed to send email: \{e\}")\
\
def load_data(file_path):\
    try:\
        df = pd.read_csv(file_path)\
        return df\
    except Exception as e:\
        send_email("Data Load Error", f"Failed to load data: \{e\}", "admin@example.com")\
        raise\
\
def clean_data(df):\
    df['personnelO'] = df['personnelO'].replace(['1 RB, 1 TE, 3 WR', '1 RB, 2 TE, 2 WR', '2 RB, 1 TE, 2 WR', '1 RB, 3 TE, 1 WR',\
                                                 '1 RB, 0 TE, 4 WR', '0 RB, 1 TE, 4 WR', '2 RB, 2 TE, 1 WR', '2 RB, 0 TE, 3 WR',\
                                                 '6 OL, 1 RB, 1 TE, 2 WR', '2 QB, 1 RB, 1 TE, 2 WR', '0 RB, 2 TE, 3 WR',\
                                                 '6 OL, 1 RB, 2 TE, 1 WR', '0 RB, 0 TE, 5 WR', '6 OL, 1 RB, 0 TE, 3 WR',\
                                                 '6 OL, 2 RB, 2 TE, 0 WR', '3 RB, 1 TE, 1 WR', '3 RB, 0 TE, 2 WR', '2 RB, 3 TE, 0 WR',\
                                                 '6 OL, 2 RB, 1 TE, 1 WR'],\
                                                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'])\
    df['personnelD'] = df['personnelD'].replace(['4 DL, 2 LB, 5 DB', '3 DL, 3 LB, 5 DB', '4 DL, 3 LB, 4 DB', '2 DL, 4 LB, 5 DB',\
                                                 '4 DL, 1 LB, 6 DB', '3 DL, 2 LB, 6 DB', '2 DL, 3 LB, 6 DB', '3 DL, 4 LB, 4 DB',\
                                                 '1 DL, 4 LB, 6 DB', '1 DL, 5 LB, 5 DB', '1 DL, 3 LB, 7 DB', '5 DL, 2 LB, 4 DB',\
                                                 '3 DL, 1 LB, 7 DB', '2 DL, 2 LB, 7 DB', '0 DL, 4 LB, 7 DB', '4 DL, 0 LB, 7 DB',\
                                                 '4 DL, 4 LB, 3 DB', '0 DL, 5 LB, 6 DB', '5 DL, 1 LB, 5 DB', '5 DL, 3 LB, 3 DB'],\
                                                ['ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au'])\
\
    processed = df[['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'yardlineNumber', 'defendersInTheBox',\
                    'numberOfPassRushers', 'preSnapVisitorScore', 'preSnapHomeScore', 'absoluteYardlineNumber',\
                    'epa', 'outcome']].copy()\
\
    cat_columns = ['offenseFormation', 'personnelD', 'personnelO']\
    processed = pd.get_dummies(df, prefix_sep="__", columns=cat_columns)\
    processed.dropna(axis=1, inplace=True)\
    processed.reset_index(drop=True, inplace=True)\
    return processed\
\
def run_models(processed):\
    data_model_y = processed.replace(\{'outcome': \{1: 'Yard gained', 0: 'No Yard gained'\}\})['outcome']\
    X_train, X_val, y_train, y_val = train_test_split(processed.loc[:, processed.columns != 'outcome'],\
                                                      data_model_y, test_size=0.7, random_state=22, stratify=processed['outcome'])\
\
    # Logistic Regression\
    log_model = LogisticRegression()\
    log_model.fit(X_train, y_train)\
    y_pred = log_model.predict(X_val)\
    log_accuracy = accuracy_score(y_val, y_pred)\
\
    # XGBoost with Randomized Search\
    xgb_model = XGBClassifier(seed=42, metric='None', n_jobs=4, silent=True)\
    param_dist = \{\
        'n_estimators': np.arange(10, 100, 10),\
        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],\
        'learning_rate': [0.05, 0.1, 0.3],\
        'subsample': stats.uniform(loc=0.2, scale=0.8),\
        'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),\
        'gamma': [0.0, 0.1, 0.2],\
        'max_depth': [5, 7, 10]\
    \}\
    rs_clf = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=50, cv=5, scoring='f1', random_state=42)\
    rs_clf.fit(X_train, y_train)\
    best_xgb_model = rs_clf.best_estimator_\
    y_pred_xgb = best_xgb_model.predict(X_val)\
    xgb_accuracy = accuracy_score(y_val, y_pred_xgb)\
\
    return log_accuracy, xgb_accuracy\
\
def main():\
    try:\
        df = load_data("plays.csv")\
        df['outcome'] = df['playResult'].apply(lambda x: 1 if x > 0 else 0)\
        processed = clean_data(df)\
        log_accuracy, xgb_accuracy = run_models(processed)\
\
        results = f"Logistic Regression Accuracy: \{log_accuracy:.4f\}\\nXGBoost Accuracy: \{xgb_accuracy:.4f\}"\
        send_email("Model Run Results", results, "admin@example.com")\
        print(results)\
    except Exception as e:\
        send_email("Pipeline Error", f"An error occurred: \{e\}", "admin@example.com")\
        raise\
\
if __name__ == "__main__":\
    main()\
}