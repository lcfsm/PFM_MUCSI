import mysql.connector
import sys

# Parámetros de conexión
DB_HOST     = 
DB_USER     = 
DB_PASSWORD = 
DB_NAME     = 

def main(fecha_str, cliente='03'):
    try:
        cnx = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
    except mysql.connector.Error as err:
        print("Error conectando a la BD:", err)
        sys.exit(1)

    cursor = cnx.cursor()

    query = """
    SELECT
      COALESCE(SUM(CASE WHEN con_tipobillete IN ('1','2','3') THEN 1 ELSE 0 END), 0) AS total_pasajeros,
      COALESCE(SUM(CASE WHEN con_tipobillete IN ('0','5','6','7','8','9') THEN 1 ELSE 0 END), 0) AS total_vehiculos
    FROM contador
    WHERE DATE(con_dateticket) = %s
      AND con_cliente = %s
    """
    cursor.execute(query, (fecha_str, cliente))
    total_pasajeros, total_vehiculos = cursor.fetchone()

    print(f"Fecha analizada    : {fecha_str}")
    print(f"Cliente            : {cliente}")
    print(f"Total pasajeros    : {total_pasajeros}")
    print(f"Total vehículos    : {total_vehiculos}")

    cursor.close()
    cnx.close()

if __name__ == '__main__':
    # Por defecto analiza el 12 de mayo de 2024 y cliente 03,
    # pero se puede pasar argumentos: python script.py YYYY-MM-DD 03
    args = sys.argv[1:]
    fecha = args[0] if len(args) > 0 else '2024-05-12'
    cliente = args[1] if len(args) > 1 else '03'
    main(fecha, cliente)

