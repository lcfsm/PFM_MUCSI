import pandas as pd

def main():
 
    csv_path = 'data/raw/dataset_empresa_03.csv'
 
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={'con_tipobillete': str}
    )
    
    df['con_dateticket'] = pd.to_datetime(
        df['con_dateticket'],
        errors='coerce'
    )
    df = df.dropna(subset=['con_dateticket'])
    df['fecha_ticket'] = df['con_dateticket'].dt.date
    
    fecha_obj = pd.to_datetime('2024-05-12').date()
    
    df_fecha = df[df['fecha_ticket'] == fecha_obj]
    
    pasajeros_vals = ['1', '2', '3']
    vehiculos_vals = ['0', '5', '6', '7', '8', '9']
    
    total_pasajeros = df_fecha['con_tipobillete'].isin(pasajeros_vals).sum()
    total_vehiculos = df_fecha['con_tipobillete'].isin(vehiculos_vals).sum()
    
    print(f"Fecha analizada    : {fecha_obj}")
    print(f"Total pasajeros    : {total_pasajeros}")
    print(f"Total vehículos    : {total_vehiculos}")

if __name__ == '__main__':
    main()
