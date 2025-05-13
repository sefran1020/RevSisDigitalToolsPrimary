import pandas as pd
import numpy as np

def analizar_todas_columnas(df_rev1, df_rev2, columnas):
    resultados = []
    for col in columnas:
        if col in df_rev1.columns and col in df_rev2.columns:
            # Simular algún cálculo de concordancia
            concordancia = np.random.rand()
            metrica = 'Kappa' if df_rev1[col].nunique() > 1 else 'Porcentaje_Acuerdo'
            resultados.append({
                'Columna': col,
                'Metrica': metrica,
                'Valor': concordancia,
                'Error_Estandar': np.random.rand() * 0.1,
                'Intervalo_Confianza': (concordancia - 0.1, concordancia + 0.1)
            })
    return resultados

def crear_df_interpretado(resultados_estandar, funcion_interpretar=None):
    if not funcion_interpretar:
        funcion_interpretar = lambda v, m: "Buena" if v > 0.6 else "Pobre"

    df = pd.DataFrame(resultados_estandar)
    if not df.empty:
        df['Interpretacion'] = df.apply(lambda row: funcion_interpretar(row['Valor'], row['Metrica']), axis=1)
        df['Metrica_Principal'] = df['Metrica']
        df['Valor_Principal'] = df['Valor']
        df['Tipo'] = 'Estándar' # Añadir tipo para la integración
    return df