# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
from pathlib import Path

def run_visualizations(model, df, reports_dir):
    """
    Генерирует графики с обратной трансформацией логарифма (Target Transformation).
    """
    sns.set_style("whitegrid")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    OFFSET = 50  # Должно совпадать с тем, что в обучении
    
    # 1. Подготовка признаков
    feature_names = model.get_booster().feature_names
    X = df[feature_names].astype(float)
    
    # 2. ПОЛУЧАЕМ ПРЕДСКАЗАНИЯ В ЛОГАРИФМАХ
    y_pred_log = model.predict(X)
    
    # 3. ОБРАТНАЯ ТРАНСФОРМАЦИЯ (Inverse Transform)
    # Это превратит "5.14" в "120.45"
    y_pred = np.expm1(y_pred_log) - OFFSET
    
    # Реальные значения (они уже должны быть в Евро в колонке 'target')
    y_true = df['target']

    # --- 1. Feature Importance ---
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    df_imp = pd.DataFrame(sorted_imp, columns=['Feature', 'Importance'])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', hue='Feature', palette='magma', ax=ax1, legend=False)
    ax1.set_title("Top 15 Features (Gain)")
    mlflow.log_figure(fig1, "plots/1_feature_importance.png")
    plt.close(fig1)

    # --- 2. Actual vs Predicted (ТЕПЕРЬ В ЕВРО) ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(y_true, y_pred, alpha=0.3, color='teal', s=10)
    # Рисуем линию идеального прогноза
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, 'r--', lw=2)
    ax2.set_xlabel("Actual Price (EUR)")
    ax2.set_ylabel("Predicted Price (EUR)")
    ax2.set_title(f"Actual vs Predicted (Real Scale)")
    mlflow.log_figure(fig2, "plots/2_actual_vs_predicted.png")
    plt.close(fig2)

    # --- 3. Time Series Zoom (Last 100h) ---
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    # Берем последние 100 точек
    plt.plot(y_true.values[-100:], label='Actual', color='black', alpha=0.8, lw=2)
    plt.plot(y_pred[-100:], label='Predicted', color='orange', linestyle='--', alpha=0.8, lw=2)
    plt.title("Price Forecast Zoom (Last 100 hours)")
    plt.ylabel("Price (EUR)")
    plt.legend()
    mlflow.log_figure(fig3, "plots/3_time_series_zoom.png")
    plt.close(fig3)
    
    print(f"Графики успешно обновлены и залогированы в MLflow (в шкале EUR).")