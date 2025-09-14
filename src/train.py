import argparse, os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow, mlflow.sklearn

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", required=True)
    p.add_argument("--out_model", required=True)  # pasta de saída (MLflow model)
    return p.parse_args()

def main():
    args = parse_args() 
    df = pd.read_csv(args.data_csv)

    X = df[["Temperatura (°C)"]]
    y = df["Vendas de Sorvete"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="regressao_sorvete"):
        model = LinearRegression().fit(Xtr, ytr)
        ypred = model.predict(Xte)
        mse, r2 = mean_squared_error(yte, ypred), r2_score(yte, ypred)

        mlflow.log_param("algoritmo", "LinearRegression")
        mlflow.log_param("feature", "Temperatura (°C)")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        os.makedirs(args.out_model, exist_ok=True)
        mlflow.sklearn.save_model(model, args.out_model)
        print(f"MSE={mse:.3f}  R2={r2:.3f}")
        print("Modelo salvo em:", args.out_model)

if __name__ == "__main__":
    main()
