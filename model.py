{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "29d9934d-c698-4398-a786-0a27b65494fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0f060f9a-097d-4537-8912-b62b79749e39",
   "metadata": {},
   "outputs": [],
   "source": [
    "dat = pd.read_csv(\"headbrain_train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "51a5d71a-7d83-410c-95e5-d2bc37a23ac8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gender                   1.0\n",
      "Age Range                1.0\n",
      "Head Size(cm^3)        487.0\n",
      "Brain Weight(grams)    143.0\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Using IQR to detect outliers\n",
    "Q1d = dat.quantile(0.25)\n",
    "Q3d = dat.quantile(0.75)\n",
    "IQRd = Q3d - Q1d\n",
    "print(IQRd)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "457cd7b4-e124-4439-be29-1b3625f0fc8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removing outliers with IQR scores\n",
    "data = dat[~((dat < (Q1d - 1.5 * IQRd)) |(dat > (Q3d + 1.5 * IQRd))).any(axis=1)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ec93db23-d2b3-4e2a-9892-50782e54ddda",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Defining variables\n",
    "x = data.iloc[:, :3]\n",
    "y = data.iloc[:, -1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "815b50d2-8682-41d5-b9a5-27a02b1291b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting training and test set\n",
    "# Since we have a small dataset, we will use all data to train the model\n",
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b90b960e-3276-41a1-8a11-9a54379eb368",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting model with training data\n",
    "regressor.fit(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2867a960-bb08-4ef4-8a5e-c5783261c81e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Saving model\n",
    "pickle.dump(regressor, open(\"model.pkl\", \"wb\"))"
   ]
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
