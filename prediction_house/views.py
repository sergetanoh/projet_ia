from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def home(request):
    return render(request,'index.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
    
        print('post')
        data = pd.read_excel("S://examen_ia//prediction//prediction_house//PROJET IA 2.xlsx")
        data.head()
        columns_to_drop = ["Observatory","agglomeration", "Zone_complementaire", "Type_habitat", "epoque_construction_homogene",
                   "anciennete_locataire_homogene", "nombre_pieces_homogene", "loyer_1_decile",
                   "loyer_1_quartile", "methodologie_production","loyer_mensuel_1_quartile","loyer_mensuel_1_decile","loyer_mensuel_median","loyer_mensuel_3_quartile","loyer_mensuel_9_decile","loyer_3_quartile","loyer_median","loyer_9_decile","loyer_moyen"]

        data.drop(columns=columns_to_drop, inplace=True)
        data.head()
        sns.heatmap(data.isnull())
        data.dropna(inplace=True)
        sns.heatmap(data.isnull())
        X=data.drop('moyenne_loyer_mensuel',axis=1)
        Y=data["moyenne_loyer_mensuel"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        anne = float(request.POST.get('annecons', 0))  # Default to 0 if 'annecons' is not provided
        surfmoy = float(request.POST.get('surfmoy', 0))
        nbreobs = float(request.POST.get('nbreobs', 0))
        nbrelog = float(request.POST.get('nbrelog', 0))

        pred = model.predict(np.array([anne, surfmoy, nbreobs, nbrelog]).reshape(1, -1))
        pred = round(pred[0])
        prix = 'Le modèle prédit une somme  :', str(pred), '€ pour les caracteristiques soumises'
        return render(request, 'predict.html', {'resultat': prix})

    # Handle the case when the form is not submitted
    
   