{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4a68a2cb-ba05-45eb-b1af-9be9a83bb5af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "import os\n",
    "from joblib import dump, load\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "\n",
    "def evaluation_pipeline(x_test_path, y_test_path, model_path):\n",
    "    x_data = pd.read_csv(x_test_path)\n",
    "    y_true=pd.read_csv(y_test_path)\n",
    "\n",
    "    loaded_scaler = load('models/scaler/min_max_scaler.pkl')\n",
    "    x_data_normalized = loaded_scaler.transform(x_data)\n",
    "    \n",
    "    loaded_model = load('models/logistic_regression.pkl')\n",
    "    y_pred = loaded_model.predict(x_data_normalized)\n",
    "    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Label'])\n",
    "    accuracy = accuracy_score(y_true, y_pred)*100\n",
    "    \n",
    "    return y_pred_df,accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83ab3bb6-8733-4525-85bd-10b78950050f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
